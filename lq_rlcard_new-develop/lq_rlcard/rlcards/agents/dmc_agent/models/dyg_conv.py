# -*- coding: utf-8

import torch
import numpy as np
import torch.nn.functional as F

from torch import nn

class ConvModel(nn.Module):
	"""
	残差网络结构(麻将血流)
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layers):
		super().__init__()

		# 输入通道数量
		self.in_planes = 4

		# 神经网络层级结构
		input_dims = np.prod(state_shape) + np.prod(action_shape)
		layer_dims = [input_dims] + mlp_layers

		# 卷积层
		self.conv1 = nn.Conv1d(self.in_planes, 4, kernel_size=(3,), stride=(2,), padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(4)

		# 残差网络层
		self.layer1 = self._make_layer(BasicBlock, 8, 2, stride=2)
		self.layer2 = self._make_layer(BasicBlock, 16, 2, stride=2)
		self.layer3 = self._make_layer(BasicBlock, 32, 2, stride=2)

		# 全连接层
		self.dense0 = nn.Linear(224, layer_dims[0])
		self.dense1 = nn.Linear(layer_dims[0], layer_dims[1])
		self.dense2 = nn.Linear(layer_dims[1], layer_dims[2])
		self.dense3 = nn.Linear(layer_dims[2], layer_dims[3])
		self.dense4 = nn.Linear(layer_dims[3], layer_dims[4])
		self.dense5 = nn.Linear(layer_dims[4], 1)

	def _make_layer(self, block, planes, num_blocks, stride):
		"""
		制作层级结构
		"""
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, z, x):
		"""
		前向传播
		"""
		# 卷积层
		out = F.relu(self.conv1(z))
		out = F.relu(self.bn1(out))

		# 残差层
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)

		out = out.flatten(1, 2)
		out = torch.cat([x, out], dim=-1)

		# 全连接层
		out = F.relu(self.dense0(out))
		out = F.relu(self.dense1(out))
		out = F.relu(self.dense2(out))
		out = F.relu(self.dense3(out))
		out = F.relu(self.dense4(out))
		out = F.relu(self.dense5(out))
		return out

	@staticmethod
	def softmax(values):
		"""
		动作概率分布
		"""
		softmax = torch.nn.Softmax(dim=0)
		return softmax(values)

class BasicBlock(nn.Module):
	"""
	用于ResNet18和34的残差块，用的是2个3x3的卷积
	"""
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,), stride=(stride,), padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,), stride=(1,), padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)
		self.shortcut = nn.Sequential()

		# 经过处理后的x要与x的维度相同(尺寸和深度)
		# 如果不相同，需要添加卷积+BN来变换为同一维度
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion * planes, kernel_size=(1,), stride=(stride,), bias=False),
				nn.BatchNorm1d(self.expansion * planes)
			)

	def forward(self, x):
		"""
		前向传播
		"""
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class DMCAgent:
	"""
	DMC代理
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layers,
			exp_epsilon=0.025,
			device="0"):
		self.use_raw = False
		self.device = 'cuda:' + device if device != "cpu" else "cpu"
		self.net = ConvModel(state_shape, action_shape, mlp_layers).to(self.device)
		self.exp_epsilon = exp_epsilon
		self.action_shape = action_shape

	def step(self, state, is_training=True):
		"""
		预测更新输出动作
		"""
		# 仅仅为一个动作时，则直接返回
		if len(state['legal_actions']) == 1:
			action_keys = np.array(list(state['legal_actions'].keys()))
			return int(action_keys)
		if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon and is_training:
			action = np.random.choice(list(state['legal_actions'].keys()))
		else:
			# 对多个动作输入到神经网络中进行预测
			action_keys, values = self.predict(state)
			action_idx = np.argmax(values)
			action = action_keys[action_idx]

		return action

	def eval_step(self, state):
		"""
		更新评估预测动作
		"""
		action_keys, values = self.predict(state)
		action_idx = np.argmax(values)
		action = action_keys[action_idx]

		info = dict()
		info['values'] = {state['raw_legal_actions'][i]: float(values[i]) for i in range(len(action_keys))}

		return action, info

	def share_memory(self):
		"""
		共享内存
		"""
		self.net.share_memory()

	def eval(self):
		"""
		评估模式
		"""
		self.net.eval()

	def parameters(self):
		"""
		神经网络参数
		"""
		return self.net.parameters()

	def predict(self, state):
		"""
		对动作进行预测
		"""
		z_obs = state['z_obs'].astype(np.float32)
		legal_actions = state['legal_actions']
		action_keys = np.array(list(legal_actions.keys()))
		action_values = list(legal_actions.values())

		# 动作特征，进行One-Hot编码
		for i in range(len(action_values)):
			if action_values[i] is None:
				action_values[i] = np.zeros(self.action_shape[0])
				action_values[i][action_keys[i]] = 1

		# 神经网络输入数据
		action_values = np.array(action_values, dtype=np.float32)
		z_obs = np.repeat(z_obs[np.newaxis, :], len(action_values), axis=0)

		# 预测Q值
		values = self.net.forward(
			torch.from_numpy(z_obs).to(self.device),
			torch.from_numpy(action_values).to(self.device),
		)

		return action_keys, values.cpu().detach().numpy()

	def forward(self, obs, actions):
		"""
		前向传递
		"""
		return self.net.forward(obs, actions)

	def load_state_dict(self, state_dict):
		"""
		加载状态字典
		"""
		return self.net.load_state_dict(state_dict)

	def state_dict(self):
		"""
		状态字典
		"""
		return self.net.state_dict()

	def set_device(self, device):
		"""
		运行设备
		"""
		self.device = device


class DMCDygModel:
	"""
	模型封装
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layers=None,
			exp_epsilon=0.01,
			device='0'):

		# 神经网络层级结构
		if mlp_layers is None:
			mlp_layers = [512, 512, 512, 512]

		# 创建DMC代理
		self.agents = []
		for player_id in range(len(state_shape)):
			agent = DMCAgent(
				state_shape[player_id],
				action_shape[player_id],
				mlp_layers,
				exp_epsilon,
				device)
			self.agents.append(agent)

	def share_memory(self):
		"""
		共享内存
		"""
		for agent in self.agents:
			agent.share_memory()

	def eval(self):
		"""
		评估模式，仅评估，不训练
		"""
		for agent in self.agents:
			agent.eval()

	def parameters(self, index):
		"""
		模型参数
		"""
		return self.agents[index].parameters()

	def get_agent(self, index):
		"""
		获取对应的模型
		"""
		return self.agents[index]

	def get_agents(self):
		"""
		获取模型
		"""
		return self.agents