# -*- coding: utf-8

import torch
import numpy as np

import torch.nn.functional as F

from torch import nn

class DMCModel(nn.Module):
	"""
	水鱼长短期记忆(模型层级)
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layer):
		super().__init__()
		input_dims = np.prod(state_shape) + np.prod(action_shape)
		layer_dims = [input_dims] + mlp_layer  # [1406, 512, 512, 512, 512]

		# 长短期记忆网络层
		# self.lstm = nn.LSTM(9, 503, batch_first=True)

		# 全连接神经网络层
		self.dense0 = nn.Linear(4424, layer_dims[0])
		self.dense1 = nn.Linear(layer_dims[0], layer_dims[1])
		self.dense2 = nn.Linear(layer_dims[1], layer_dims[2])
		self.dense3 = nn.Linear(layer_dims[2], layer_dims[3])
		self.dense4 = nn.Linear(layer_dims[3], layer_dims[4])
		self.dense5 = nn.Linear(layer_dims[4], layer_dims[5])
		self.dense6 = nn.Linear(layer_dims[5], 1)

	def forward(self, obs, actions):
		"""
		前向传播
		"""
		obs = torch.flatten(obs, 1)
		actions = torch.flatten(actions, 1)

		# 长短期记忆网络层
		# lstm_out, (h_n, _) = self.lstm(actions)
		x = torch.cat([obs, actions], dim=1)

		# 全连接层
		x = F.leaky_relu_(self.dense0(x))
		x = F.leaky_relu_(self.dense1(x))
		x = F.leaky_relu_(self.dense2(x))
		x = F.leaky_relu_(self.dense3(x))
		x = F.leaky_relu_(self.dense4(x))
		x = F.leaky_relu_(self.dense5(x))
		x = F.leaky_relu_(self.dense6(x))
		# x = (x - x.mean()) / (x.std() + 1e-8)
		values = self.softmax(x)

		return values

	@staticmethod
	def softmax(values):
		"""
		动作概率分布
		"""
		softmax = torch.nn.Softmax(dim=0)
		return softmax(values)


class DMCLandlordModel(nn.Module):
	"""
	水鱼长短期记忆(模型层级)
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layer):
		super().__init__()
		input_dims = np.prod(state_shape) + np.prod(action_shape)
		layer_dims = [input_dims] + mlp_layer  # [1406, 512, 512, 512, 512]

		# 长短期记忆网络层
		self.lstm = nn.LSTM(9, 503, batch_first=True)

		# 全连接神经网络层
		self.dense0 = nn.Linear(4424, layer_dims[0])
		self.dense1 = nn.Linear(layer_dims[0], layer_dims[1])
		self.dense2 = nn.Linear(layer_dims[1], layer_dims[2])
		self.dense3 = nn.Linear(layer_dims[2], layer_dims[3])
		self.dense4 = nn.Linear(layer_dims[3], layer_dims[4])
		self.dense5 = nn.Linear(layer_dims[4], layer_dims[5])
		self.dense6 = nn.Linear(layer_dims[5], 1)

	def forward(self, obs, actions):
		"""
		前向传播
		"""
		obs = torch.flatten(obs, 1)
		actions = torch.flatten(actions, 1)

		# 长短期记忆网络层
		lstm_out, (h_n, _) = self.lstm(actions)
		x = torch.cat([obs, actions], dim=1)

		# 全连接层
		x = F.leaky_relu_(self.dense0(x))
		x = F.leaky_relu_(self.dense1(x))
		x = F.leaky_relu_(self.dense2(x))
		x = F.leaky_relu_(self.dense3(x))
		x = F.leaky_relu_(self.dense4(x))
		x = F.leaky_relu_(self.dense5(x))
		x = F.leaky_relu_(self.dense6(x))
		# x = (x - x.mean()) / (x.std() + 1e-8)
		values = self.softmax(x)

		return values

	@staticmethod
	def softmax(values):
		"""
		动作概率分布
		"""
		softmax = torch.nn.Softmax(dim=0)
		return softmax(values)


class DMCFarmerModel(nn.Module):
	"""
	水鱼长短期记忆(模型层级)
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layer):
		super().__init__()
		input_dims = np.prod(state_shape) + np.prod(action_shape)
		layer_dims = [input_dims] + mlp_layer  # [1406, 512, 512, 512, 512]

		# 长短期记忆网络层
		self.lstm = nn.LSTM(9, 503, batch_first=True)

		# 全连接神经网络层
		self.dense0 = nn.Linear(4424, layer_dims[0])
		self.dense1 = nn.Linear(layer_dims[0], layer_dims[1])
		self.dense2 = nn.Linear(layer_dims[1], layer_dims[2])
		self.dense3 = nn.Linear(layer_dims[2], layer_dims[3])
		self.dense4 = nn.Linear(layer_dims[3], layer_dims[4])
		self.dense5 = nn.Linear(layer_dims[4], layer_dims[5])
		self.dense6 = nn.Linear(layer_dims[5], 1)

	def forward(self, obs, actions):
		"""
		前向传播
		"""
		obs = torch.flatten(obs, 1)
		actions = torch.flatten(actions, 1)

		# 长短期记忆网络层
		lstm_out, (h_n, _) = self.lstm(actions)
		x = torch.cat([obs, actions], dim=1)

		# 全连接层
		x = F.leaky_relu_(self.dense0(x))
		x = F.leaky_relu_(self.dense1(x))
		x = F.leaky_relu_(self.dense2(x))
		x = F.leaky_relu_(self.dense3(x))
		x = F.leaky_relu_(self.dense4(x))
		x = F.leaky_relu_(self.dense5(x))
		x = F.leaky_relu_(self.dense6(x))
		# x = (x - x.mean()) / (x.std() + 1e-8)
		values = self.softmax(x)

		return values

	@staticmethod
	def softmax(values):
		"""
		动作概率分布
		"""
		softmax = torch.nn.Softmax(dim=0)
		return softmax(values)


class DMCAgent:
	"""
	DMC代理
	"""
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layers,
			exp_epsilon=0.01,
			device="0"):
		self.use_raw = False
		self.device = 'cuda:' + device if device != "cpu" else "cpu"
		self.net = DMCModel(state_shape, action_shape, mlp_layers).to(self.device)
		self.exp_epsilon = exp_epsilon
		self.action_shape = action_shape

	def step(self, state, is_training=True):
		"""
		预测更新输出动作
		"""
		legal_actions = state['actions']
		# 仅存在一个合法动作时，直接返回
		if len(legal_actions) == 1:
			action_keys = np.array(list(legal_actions.keys()))
			return int(action_keys)
		if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon and is_training:
			# 随机选取
			action = np.random.choice(list(legal_actions.keys()))
		else:
			# 神经网络预测(概率最大的动作)
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
		""" 共享内存"""
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
		obs = state['obs'].astype(np.float32)
		legal_actions = state['actions']
		action_keys = np.array(list(legal_actions.keys()))
		action_values = list(legal_actions.values())

		# 动作值为None时，则进行动作编码One-hot
		for i in range(len(action_values)):
			if action_values[i] is None:
				action_values[i] = np.zeros(self.action_shape[0])
				action_values[i][action_keys[i]] = 1
		action_values = np.array(action_values, dtype=np.float32)
		obs = np.repeat(obs[np.newaxis, :], len(action_keys), axis=0)

		# 网络预测概率分布
		values = self.net.forward(
			torch.from_numpy(obs).to(self.device),
			torch.from_numpy(action_values).to(self.device)
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


class Model:
	def __init__(
			self,
			state_shape,
			action_shape,
			mlp_layers=None,
			exp_epsilon=0.01,
			device='0'):

		# 神经网络层级结构
		if mlp_layers is None:
			mlp_layers = [512, 512, 512, 512, 14]

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
		设置为评估模式，不进行训练
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