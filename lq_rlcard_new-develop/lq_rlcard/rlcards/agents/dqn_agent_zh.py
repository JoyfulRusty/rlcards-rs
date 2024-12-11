# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class DQNAgent(object):
	'''
	DQNAgent 玩家代理
	'''
	def __init__(
			self,
			replay_memory_size = 20000,
			replay_memory_init_size = 100,
			update_target_estimator_every = 1000,
			discount_factor = 0.99,
			epsilon_start = 1.0,
			epsilon_end = 0.1,
			epsilon_decay_steps = 20000,
			batch_size = 32,
			num_actions = 2,
			state_shape = None,
			train_every = 1,
			mlp_layers = None,
			learning_rate = 0.00005,
			device = None):
		'''
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
		'''
		self.use_raw = False
		self.replay_memory_init_size = replay_memory_init_size
		self.update_target_estimator_every = update_target_estimator_every
		self.discount_factor = discount_factor
		self.epsilon_decay_steps = epsilon_decay_steps
		self.batch_size = batch_size
		self.num_actions = num_actions
		self.train_every = train_every

		# CPU or GPU
		if device is None:
			self.device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device
		# 总共时间步长
		self.total_t = 0
		# 总共训练时间步长
		self.train_t = 0
		# ε衰变调度器 (1., 0.1, 20000)
		self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

		# TODO: 评估与评测
		# 创建评估器
		self.q_estimator = Estimator(
			num_actions = num_actions,
			learning_rate = learning_rate,
			state_shape = state_shape,
			mlp_layers = mlp_layers,
			device = self.device)

		# 创建目标估计器
		self.target_estimator = Estimator(num_actions = num_actions,
										  learning_rate = learning_rate,
										  state_shape = state_shape,
										  mlp_layers = mlp_layers,
										  device = self.device)
		# 创建经验回放记忆内存
		self.memory = Memory(replay_memory_size, batch_size)

	def feed(self, ts):
		'''
		注入数据，将数据存储到重放缓冲区并训练代理，有两个阶段:
			1.在第1阶段，在没有训练的情况下填充内存
            2.在第2阶段，每隔几个时间步培训一次代理
		:param ts: 代表过渡的5个元素的列表
		'''
		# print('ts: ', ts)
		(state, action, reward, next_state, done) = tuple(ts)
		# 执行梯度下降更新，馈送到内存
		self.feed_memory(
			state['obs'],
			action,
			reward,
			next_state['obs'],
			list(next_state['legal_actions'].keys()),
			done)
		# 时间步长记录+1
		self.total_t += 1
		tmp = self.total_t - self.replay_memory_init_size
		if tmp >= 0 and tmp % self.train_every == 0:
			self.train()

	def step(self, state):
		'''
		预测生成数据的行动，但将预测与计算图进行断开
		:param state: 当前状态
		:return: action(int): 操作id
		'''
		# Q值
		q_values = self.predict(state)
		# ε衰变值
		epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
		# 合法动作列表
		legal_actions = list(state['legal_actions'].keys())
		prob_s = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
		# 最好的动作idx
		best_action_idx = legal_actions.index(np.argmax(q_values))
		prob_s[best_action_idx] += (1.0 - epsilon)
		action_idx = np.random.choice(np.arange(len(prob_s)),  p = prob_s)

		return legal_actions[action_idx]

	def eval_step(self, state):
		'''
		评估步骤，预测用于评估的操作
		:param state: 当前状态
		:return: action(int): 动作id, info(dict): 包含信息的字典
		'''
		# 预测Q值
		q_values= self.predict(state)
		# 最好的动作
		best_action = np.argmax(q_values)
		# 数据信息
		info = {}
		info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]])
						  for i in range(len(state['legal_actions']))}

		return best_action, info

	def predict(self, state):
		'''
		TODO: predict和train中使用的是同一个网络，网络传入参数存在区别
		predict预测masked_q_values(掩码Q值)
		:param state: 当前状态
		:return: q_values(np.array): 一个一维数组，其中每个条目代表一个Q值
		'''
		# Q值
		q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0] # 展开数组的形状
		# 掩码Q值
		masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float) # np.inf 表示+∞，没有确切的数值的,类型为浮点型
		# 合法动作
		legal_actions = list(state['legal_actions'].keys())
		# 掩码Q值中合法动作
		masked_q_values[legal_actions] = q_values[legal_actions]

		return masked_q_values

	def train(self):
		'''
		TODO: 训练网络，每次训练的时候从经验回放池中采集
		:return: loss(float): 当前批次的损失值
		'''
		# 从经验回放池中采集对应的动作(参数)
		# print('self.memory.sample: ', self.memory.sample())
		state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()
		# 使用Q-Network计算最佳下一步行动(下一个Q值动作)
		q_values_next= self.q_estimator.predict_nograd(next_state_batch)

		# 合法动作
		legal_actions = []
		for b in range(self.batch_size):
			# legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch][b])
			legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
		# 掩码动作
		masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype = float)
		masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
		masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))

		# 最好的动作
		best_actions = np.argmax(masked_q_values, axis = 1)
		# 使用目标网络评估下一个最佳动作(下一步Q值)
		q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
		# 目标网络批次
		target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * \
					   q_values_next_target[np.arange(self.batch_size), best_actions]
		# 执行梯度下降更新
		state_batch = np.array(state_batch)
		# 计算Loss损失值
		loss = self.q_estimator.update(state_batch, action_batch, target_batch)
		print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

		# 更新目标估计器
		if self.train_t % self.update_target_estimator_every == 0:
			self.target_estimator = deepcopy(self.q_estimator)
			print("\nINFO - Copied models parameters to target network")

		self.train_t += 1

	def feed_memory(self, state, action, reward, next_state, legal_action, done):
		'''
		执行梯度下降更新，馈送到内存
        :param state(np.array): 当前状态
        :param action(int): 执行操作的ID
        :param reward(np.array): 收到的奖励
        :param next_state: 执行操作后的下一个状态
        :param legal_actions(list): 下一个合法动作
        :param done(boolean): 是否结束
		'''
		self.memory.save(state, action, reward, next_state, legal_action, done)

	def set_device(self, device):
		'''
		设定计算设备
		:param divice: CPU or GPU
		'''
		self.device = device
		# 评估
		self.q_estimator.device = device
		# 目标
		self.target_estimator.device = device


class Estimator(object):
	'''
	Q值估计器神经网络，该网络用于Q网络和目标网络
	'''
	def __init__(self, num_actions = 2, learning_rate = 0.001, state_shape = None, mlp_layers = None, device = None):
		'''
		初始化Estimator属性参数
		:param num_actions: 输出操作数
		:param learning_rate: 学习效率
		:param state_shape: 状态空间的形状
		:param mlp_layers: mlp层输出的大小
		:param device: CPU or GPU
		'''
		self.num_actions = num_actions
		self.learning_rate = learning_rate
		self.state_shape = state_shape
		self.mlp_layers = mlp_layers
		self.device = device

		# 设置Q网络，并将其设定为eval模式
		q_net = EstimatorNetwork(num_actions, state_shape, mlp_layers)
		q_net = q_net.to(device)
		self.q_net = q_net
		self.q_net.eval()

		# 使用Xavier init初始化权重
		for p in self.q_net.parameters():
			if len(p.data.shape) > 1:
				nn.init.xavier_normal_(p.data) # # xavier_normal_ 初始化

		# 设置损失函数
		self.mes_loss = nn.MSELoss(reduction = 'mean')
		# 设置优化器
		self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = self.learning_rate)

	def predict_nograd(self, s):
		'''
		TODO: 预测操作值，但不包括预测在计算图中的值，用于预测最佳的下一步，DQN中的操作
		:param s(np.array): 批处理(state len)
		:return: np.array(batch_size, NUM_VALID_ACTIONS), 包含估计的动作值
		'''
		# 需要进行反向传播，因为没有使用到计算图
		with torch.no_grad():
			s = torch.from_numpy(s).float().to(self.device)
			q_value = self.q_net(s).cpu().numpy()
		return q_value

	def update(self, s, a, y):
		'''
		TODO: 向给定目标更新估计器，在这种情况下，y是估计目标网络，Q网络预测最佳动作值
		:param s:
		:param a:
		:param y:
		:return: 计算批次损失值
		'''
		# 零梯度下降
		self.optimizer.zero_grad()
		self.q_net.train()
		s = torch.from_numpy(s).float().to(self.device)
		a = torch.from_numpy(a).long().to(self.device)
		y = torch.from_numpy(y).float().to(self.device)

		# # (batch, state_shape) -> (batch, num_actions)
		q_value = self.q_net(s)
		# (batch, num_actions) -> (batch, )
		Q = torch.gather(q_value, dim = -1, index = a.unsqueeze(-1)).squeeze(-1) # # gather: 利用index来索引input特定位置的数值
		# 更新网络模型
		batch_loss = self.mes_loss(Q, y)
		batch_loss.backward()

		self.optimizer.step()
		batch_loss = batch_loss.item()

		return batch_loss


class EstimatorNetwork(nn.Module):
	'''
	TODO: 估计器函数逼近网络
	tanh层，所有输入/输出都是torch.tensor
	'''
	def __init__(self, num_actions = 2, state_shape = None, mlp_layers = None):
		'''
		初始化EstimatorNetwork属性参数
		:param num_actions: 合法动作次数
		:param state_shape: 状态张量形状
		:param mlp_layers: 每个全连接层fc输出大小
		'''
		super(EstimatorNetwork, self).__init__()
		self.num_actions = num_actions
		self.state_shape = state_shape
		self.mlp_layers = mlp_layers

		# 构建EstimatorNetwork网络
		# np.prod: 默认情况下，返回定轴上的数组元素的乘积，即计算所有元素的乘积
		layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
		fc = [nn.Flatten()]
		fc.append(nn.BatchNorm1d(layer_dims[0]))
		for i in range(len(layer_dims) - 1):
			fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
			fc.append(nn.Tanh())

		fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias = True))
		self.fc_layers = nn.Sequential(*fc)

	def forward(self, s):
		'''
		预测操作值
		:param s(Tensor): (批处理，状态，形状)
		'''
		return self.fc_layers(s)


class Memory(object):
	'''
	TODO: 用于保存和转换内存数据
	'''
	def __init__(self, memory_size, batch_size):
		'''
		初始化内存参数(经验回放池)
		:param memory_size: 内存大小
		:param batch_size: 批次大小
		'''
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.memory = []  # 内存大小

	def save(self, state, action, reward, next_state, legal_actions, done):
		'''
		TODO: 保存transition到memory
		:param state(numpy.array): 当前状态
		:param action(int): 动作ID
		:param reward(float): 奖励
		:param next_state(numpy.array): 下一个动作
		:param legal_action(list): 下一个合法动作
		:param done(boolean): 是否结束
		'''
		if len(self.memory) == self.memory_size:
			self.memory.pop(0)
		transition = Transition(state, action, reward, next_state, done, legal_actions)
		self.memory.append(transition)

	def sample(self):
		'''
		从经验回放池中进行最小批次的采样
		:return:
			state_batch(list): 批次状态
			action_batch(list): 动作批次
			reward_batch(list): 奖励批次
			next_state_batch(list): 下一个状态批次
			done_batch(list): 是否结束批次
		'''
		# samples = random.sample(self.memory, self.batch_size)
		# return map(np.array, zip(*samples))
		samples = random.sample(self.memory, self.batch_size)
		samples = tuple(zip(*samples))
		print('samples: ', samples)
		# print('samples[:-1]: ', samples[:-1])
		print('#' * 20)
		print('samples[-1]: ', samples[-1])
		return tuple(map(np.array, samples[:-1])) + (samples[-1],)