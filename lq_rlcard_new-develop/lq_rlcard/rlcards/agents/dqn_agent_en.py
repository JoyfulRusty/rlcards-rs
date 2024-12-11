# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class DQNAgent(object):
	"""
	DQNAgent代理
	"""

	def __init__(
			self,
			replay_memory_size=20000,
			replay_memory_init_size=100,
			update_target_estimator_every=1000,
			discount_factor=0.99,
			epsilon_start=1.0,
			epsilon_end=0.1,
			epsilon_decay_steps=20000,
			batch_size=32,
			num_actions=2,
			state_shape=None,
			train_every=1,
			mlp_layers=None,
			learning_rate=0.00005,
			device=None,
			save_path=None,
			save_every=float('inf'), ):
		"""
		使用函数逼近的非策略TD控制的Q-学习算法, 在遵循epsilon贪婪策略的同时找到最优贪婪策略
		:param replay_memory_size: 缓存记忆大小
		:param replay_memory_init_size: 初始化时，要采样的随机体验数回复存储器
		:param update_target_estimator_every: 将参数从Q估计器复制到每N步目标估计器
		:param discount_factor: 伽马折扣因子
		:param epsilon_start: 在执行动作时，对随机动作进行采样的机会， epsilon随时间衰减，这是开始值
		:param epsilon_end: 衰减完成后epsilon的最终值的最小值
		:param epsilon_decay_steps: 衰减epsilon的步数
		:param batch_size: 要重放内存中采样的批次大小
		:param evaluate_every: 每N步求值一次
		:param num_actions: 操作数
		:param state_shape: 状态向量空间shape
		:param train_every: 每X步，训练一次网络
		:param mlp_layers: mlp中每个层的层号和维度
		:param learning_rate: DQN代理的学习效率
		:param device: cpu or gpu
		"""
		self.use_raw = False
		self.replay_memory_init_size = replay_memory_init_size
		self.update_target_estimator_every = update_target_estimator_every
		self.discount_factor = discount_factor
		self.epsilon_decay_steps = epsilon_decay_steps
		self.batch_size = batch_size
		self.num_actions = num_actions
		self.train_every = train_every

		# CUDA OR CPU
		if device is None:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device

		# 总的时间步长
		self.total_t = 0

		# 总的训练步长
		self.train_t = 0

		# ε衰变调度器
		self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

		# 创建Q估计器和目标网络估计器
		self.q_estimator = Estimator(
			num_actions=num_actions,
			learning_rate=learning_rate,
			state_shape=state_shape,
			mlp_layers=mlp_layers,
			device=self.device)

		self.target_estimator = Estimator(
			num_actions=num_actions,
			learning_rate=learning_rate,
			state_shape=state_shape,
			mlp_layers=mlp_layers,
			device=self.device)

		# 创建缓存记忆
		self.memory = Memory(replay_memory_size, batch_size)

		# 检查保存参数
		self.save_path = save_path
		self.save_every = save_every

	def feed(self, ts):
		"""
		注入数据，将数据存储到重放缓冲区并训练代理，分为两个阶段：
			1.在第1阶段，在没有训练的情况下填充内存
			2.在第2阶段，每隔几个时间步培训一次代理
			3.ts，代表过渡的5个元素的列表
		"""
		# 执行梯度更新，馈送到内存
		(state, action, reward, next_state, done) = tuple(ts)
		# 执行梯度更新，馈送到内存
		if isinstance(action, int):
			self.feed_memory(
				state['obs'],
				action,
				reward,
				next_state['obs'],
				list(next_state['actions'].keys()),
				done)

		# 计算总的时间步长
		# self.total_t == 100, tmp >= 0
		# 则需要经过100次的时间步长
		self.total_t += 1
		tmp = self.total_t - self.replay_memory_init_size
		if tmp >= 0 and tmp % self.train_every == 0:
			# print(f'玩家{i}轨迹进行训练~~~')
			self.train()

	def step(self, state):
		"""
		预测生成数据动作，但预测与计算图断开连接
		"""
		# 预测出Q值
		q_values = self.predict(state)
		# 衰变调度
		epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
		# 合法动作
		legal_actions = list(state['actions'].keys())
		# 计算合法动作
		prob = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
		best_action_idx = legal_actions.index(np.argmax(q_values))
		prob[best_action_idx] += (1.0 - epsilon)
		action_idx = np.random.choice(np.arange(len(prob)), p=prob)

		return legal_actions[action_idx]

	def predict_step(self, state):
		"""
		预测动作
		"""
		# 预测Q值
		q_values = self.predict(state)
		# 获取合法动作
		legal_actions = list(state['actions'].keys())
		# 选择最好的动作索引id
		best_action_idx = legal_actions.index(np.argmax(q_values))
		return legal_actions[best_action_idx]

	def eval_step(self, state):
		"""
		评估步骤，预测用于评估操作
		"""
		# 创建一个字典容器
		q_values = self.predict(state)
		# 评估预测的Q值
		best_action = np.argmax(q_values)
		# 抽取最好的动作
		info = {}

		info['values'] = {
			state['row_legal_actions'][i]:  float(q_values[list(state['actions'].keys())[i]]) for i in range(len(state['actions']))}

		return best_action, info

	def predict(self, state):
		"""
		预测masked_q_values(掩码Q值)
		q_values为一个一维数组，其中每个条目代表一个Q值
		"""
		# Q值
		q_values = self.q_estimator.predict_no_grad(np.expand_dims(state['obs'], 0))[0]  # expand_dim, 展开数组的形状
		# 计算掩码Q值
		masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)  # np.inf 表示+∞，没有确切的数值的,类型为浮点型
		# 获取合法动作的keys
		legal_actions = list(state['actions'].keys())
		# 捡牌
		masked_q_values[legal_actions] = q_values[legal_actions]

		return masked_q_values

	def train(self):
		"""
		TODO: 训练网络，每次训练的时候从经验回放池中采集
		"""
		# 从经验回放池采集对应的批次参数
		state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch  = self.memory.sample()

		# 使用Q—Network计算最佳的下一步动作(下一个Q值动作)
		# 通过计算下一个状态批次，预测下一个出牌动作
		q_values_next = self.q_estimator.predict_no_grad(next_state_batch)
		legal_actions = []
		for b in range(self.batch_size):
			legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
		masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
		masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
		masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
		best_actions = np.argmax(masked_q_values, axis=1)

		# 使用目标网络评估下一个最佳动作(下一步Q值)
		q_values_next_target = self.target_estimator.predict_no_grad(next_state_batch)
		target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
					   self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

		# 转换状态批次类型为数组
		state_batch = np.array(state_batch)

		# 计算损失函数
		loss = self.q_estimator.update(state_batch, action_batch, target_batch)
		print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

		# 更新目标估计器
		if self.train_t % self.update_target_estimator_every == 0:
			self.target_estimator = deepcopy(self.q_estimator)
			print("\nINFO - Copied model parameters to target network.")

		self.train_t += 1

	def feed_memory(self, state, action, reward, next_state, legal_actions, done):
		"""
		执行梯度下降更新，馈送到内存
		"""
		self.memory.save(state, action, reward, next_state, legal_actions, done)

	def set_device(self, device):
		"""
		设置运行设备，CUDA or CPU
		"""
		self.device = device
		self.q_estimator.device = device
		self.target_estimator.device = device

	def checkpoint_attributes(self):
		"""
		检查点属性
		"""

		return {
			'agent_type': 'DQNAgent',
			'q_estimator': self.q_estimator.checkpoint_attributes(),
			'memory': self.memory.checkpoint_attributes(),
			'total_t': self.total_t,
			'train_t': self.train_t,
			'replay_memory_init_size': self.replay_memory_init_size,
			'update_target_estimator_every': self.update_target_estimator_every,
			'discount_factor': self.discount_factor,
			'epsilon_start': self.epsilons.min(),
			'epsilon_end': self.epsilons.max(),
			'epsilon_decay_steps': self.epsilon_decay_steps,
			'batch_size': self.batch_size,
			'num_actions': self.num_actions,
			'train_every': self.train_every,
			'device': self.device,
			'save_path': self.save_path,
			'save_every': self.save_every
		}

	@classmethod
	def from_checkpoint(cls, checkpoint):
		"""
		从检查点恢复模型
		"""
		print("\nINFO - Restoring model from checkpoint...")
		agent_instance = cls(
			replay_memory_size=checkpoint['memory']['memory_size'],
			replay_memory_init_size=checkpoint['replay_memory_init_size'],
			update_target_estimator_every=checkpoint['update_target_estimator_every'],
			discount_factor=checkpoint['discount_factor'],
			epsilon_start=checkpoint['epsilon_start'],
			epsilon_end=checkpoint['epsilon_end'],
			epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
			batch_size=checkpoint['batch_size'],
			num_actions=checkpoint['num_actions'],
			state_shape=checkpoint['q_estimator']['state_shape'],
			train_every=checkpoint['train_every'],
			mlp_layers=checkpoint['q_estimator']['mlp_layers'],
			learning_rate=checkpoint['q_estimator']['learning_rate'],
			device=checkpoint['device'],
			save_path=checkpoint['save_path'],
			save_every=checkpoint['save_every'],
		)

		agent_instance.total_t = checkpoint['total_t']
		agent_instance.train_t = checkpoint['train_t']

		agent_instance.q_estimator = Estimator.from_checkpoint(checkpoint['q_estimator'])
		agent_instance.target_estimator = deepcopy(agent_instance.q_estimator)
		agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])

		return agent_instance

	def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
		"""
		保存模型检查点
		"""
		torch.save(self.checkpoint_attributes(), os.path.join(path, filename))

class Estimator(object):
	"""
	Q值估计器，该网络用于Q网络和目标网络
	"""
	def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
		"""
		初始化Estimator属性参数
		"""
		self.num_actions = num_actions
		self.learning_rate = learning_rate
		self.state_shape = state_shape
		self.mlp_layers = mlp_layers
		self.device = device

		# 设置Q网络并将其设定为eval模式
		qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
		qnet = qnet.to(self.device)
		self.qnet = qnet
		self.qnet.eval()

		# 使用Xavier init初始化权重
		for p in self.qnet.parameters():
			if len(p.data.shape) > 1:
				nn.init.xavier_uniform_(p.data)

		# 损失函数
		self.mse_loss = nn.MSELoss(reduction='mean')

		# 优化器
		self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

	def predict_no_grad(self, s):
		"""
		TODO: 预测操作值，但不包括预测在计算图中的值，用于预测最佳的下一步，DQN中的操作
		torch.no_grad()就是一个循环，其中该循环中的每个张量都将requires_grad设置为False。
		这意味着当前附加到当前计算图的梯度张量现在与当前图分离，将不再能够计算与该张量的梯度。
		直到张量在循环内，它才会与当前图分离。
		一旦用梯度定义的张量脱离循环，它就会再次附加到当前图。
		此方法禁用梯度计算，从而减少计算的内存消耗。
		"""
		with torch.no_grad():
			s = torch.from_numpy(s).float().to(self.device)
			q_as = self.qnet(s).cpu().numpy()
		return q_as

	def update(self, s, a, y):
		"""
		TODO: 向给定目标更新估计器，在这种情况下，y是估计目标网络，Q网络预测最佳动作值
		根据pytorch中backward()函数的计算，当网络参量进行反馈时，梯度是累积计算而不是被替换，
		但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，
		因此需要对每个batch调用一遍zero_grad()将参数梯度置0.

		另外，如果不是处理每个batch清除一次梯度，而是两次或多次再清除一次，相当于提高了batch_size，
		对硬件要求更高，更适用于需要更高batch_size的情况。
		"""
		self.optimizer.zero_grad()

		self.qnet.train()

		s = torch.from_numpy(s).float().to(self.device)
		a = torch.from_numpy(a).long().to(self.device)
		y = torch.from_numpy(y).float().to(self.device)

		q_as = self.qnet(s)
		Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

		batch_loss = self.mse_loss(Q, y)
		batch_loss.backward()
		self.optimizer.step()
		batch_loss = batch_loss.item()

		self.qnet.eval()

		return batch_loss

	def checkpoint_attributes(self):
		"""
		返回从检查点恢复模型所需的属性
		"""
		return {
			'qnet': self.qnet.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'num_actions': self.num_actions,
			'learning_rate': self.learning_rate,
			'state_shape': self.state_shape,
			'mlp_layers': self.mlp_layers,
			'device': self.device
		}

	@classmethod
	def from_checkpoint(cls, checkpoint):
		"""
		从检查点恢复模型
		"""
		estimator = cls(
			num_actions=checkpoint['num_actions'],
			learning_rate=checkpoint['learning_rate'],
			state_shape=checkpoint['state_shape'],
			mlp_layers=checkpoint['mlp_layers'],
			device=checkpoint['device']
		)

		estimator.qnet.load_state_dict(checkpoint['qnet'])
		estimator.optimizer.load_state_dict(checkpoint['optimizer'])
		return estimator

class EstimatorNetwork(nn.Module):
	"""
	TODO: 估计器函数逼近网络， tanh层，所有输入/输出都是torch.tensor
	"""
	def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
		"""
		初始化网络的属性和参数
		"""
		super(EstimatorNetwork, self).__init__()

		self.num_actions = num_actions
		self.state_shape = state_shape
		self.mlp_layers = mlp_layers

		# build the Q network
		layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
		fc = [nn.Flatten()]
		fc.append(nn.BatchNorm1d(layer_dims[0]))
		for i in range(len(layer_dims) - 1):
			fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
			fc.append(nn.Tanh())
		fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
		self.fc_layers = nn.Sequential(*fc)

	def forward(self, s):
		"""
		预测动作值，反向传递
		s(Tensor): (batch, state_shape)
		"""
		return self.fc_layers(s)

class Memory(object):
	"""
	用于保存转换的内存
	"""
	def __init__(self, memory_size, batch_size):
		"""
		初始化缓冲内存大小
		"""
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.memory = []

	def save(self, state, action, reward, next_state, legal_actions, done):
		"""
		TODO: 保存transition到memory
		"""
		if len(self.memory) == self.memory_size:
			self.memory.pop(0)
		transition = Transition(state, action, reward, next_state, done, legal_actions)
		self.memory.append(transition)

	def sample(self):
		"""
		从经验回放池中进行最小批次的采样
		"""
		samples = random.sample(self.memory, self.batch_size)
		samples = tuple(zip(*samples))
		return tuple(map(np.array, samples[:-1])) + (samples[-1],)

	def checkpoint_attributes(self):
		"""
		返回需要检查点的属性
		"""
		return {
			'memory_size': self.memory_size,
			'batch_size': self.batch_size,
			'memory': self.memory
		}

	@classmethod
	def from_checkpoint(cls, checkpoint):
		"""
		从检查点恢复属性
		"""
		instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
		instance.memory = checkpoint['memory']
		return instance