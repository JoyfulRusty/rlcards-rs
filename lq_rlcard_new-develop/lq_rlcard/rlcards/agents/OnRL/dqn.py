# -*- coding: utf-8 -*-

# todo: DQN算法其主要思想是用一个神经网络来表示最优策略的函数，然后利用Q-learning的思想进行参数更新。为了保证训练的稳定性和高效性，
#       DQN算法引入了经验回放和目标网络两大模块，使得算法在实际应用时能够取得更好的效果
'''
回顾Q-learning中的Q表:
以矩阵的方式建立了一张存储每个状态下所有动作Q值的表格，表格中的每个动作价值Q(s, a)表示在状态s下选取动作a然后继续遵顼某一策略预期能够得到的期望回报
这种用表格存储动作价值的做法只在环境的状态和动作哦都是离散的，并且空间都比较小的情况下适用，当动作或者数量非常大的时候，这样的做法很鸡肋，当状态数据
特别大的时候，在计算中中存储这个数量级的Q值表格不现实，当状态或者动作连续的时候，就有无限个状态动作对，导致更加无法使用这种表格形式来记录各个状态动作对的Q值

由于状态每一维度的值都是连续的，无法使用表达记录，常见的处理方式是使用函数拟合的思想，由于神经网络具有强大的表达能力，因此可使用一个神经网络来表示函数Q
若动作是连续无限的，神经网络的输入是状态s和动作a，然后输出一个标量，表示在状态s下采取动作a能够获取的价值，若动作是离散有限的，除了可采取动作连续情况
的做法，还可将状态s输入到神经网络中，使其输出每个动作的Q值，通常DQN以及Q-learning只能处理离散的情况，因为在函数Q的更新过程中由max_a这一操作，假设
神经网络用来拟合函数w的参数是: 即每一个状态s下所有可能动作a的Q值都能表示为Qw(s, a)，用于拟合函数Q函数的神经网络称为Q网络

在一般的有监督学习总，假设训练数据是独立同分布的，每次训练神经网络的时候从训练数据中随机采样一个或若干个数据来进行梯度下降，随学习的不断进行，每个训练
数据会被使用多次，在原来Q-learning算法中，每个数据只会用来更新一次Q值，为了更好的将Q-learning和深度神经网络结合，DQN采用了经验回放(experience
replay方法)，具体做法: 维护一个回放缓冲区，将每次采样得到的四元组(状态，动作，奖励，下一个状态)存储到回放缓存区中，训练Q网络时，再从回放缓冲区中随机
采样若干个数据来进行训练

采样具体作用:
1.使样本满足独立假设，在MDP中交互采样得到的数据本身不满足独立假设，因为这一时刻的状态和上一时刻的状态有关，非独立同分布的数据对训练神经网络有很大的影响
会使神经网络拟合到最近训练的数据上，采样经验回放可打破样本之间的相关性，使其满足独立假设

2.提高样本效率，每一个样本可用被使用多次，十分适合深度神经网络的梯度学习


Q-learning的更新规则: Q(s, a) = Q(s, a) + α[r + γ * maxQ(s`, a`) - Q(s, a)]

上述Q-learning公式使用时序差分(temporal difference, TD)学习目标[r+γ * maxQ(s`, a`)]，来增量更新Q(s, a)，也就是说要使Q(s, a)和TD目标
[r+γ * maxQ(s`, a`)]，于是，对于一组数据{(si, ai, ri, s`i)}，可将Q网络的损失函数构造为均方误差的形式:
	w` = arg_min(1/2N * 累加求和([Qw(si, ai) - (ri + γ * max(Qw(s`, a`)))] ** 2)

根据当前网络Qw(s, a)以£-greedy贪婪策略选择动作at, 执行动作at，获得回报rt，环境状态变为st+1，将(st, at, rt, st+1)存储到经验回放池R中，若R中
数据足够，从R中采样N个数据{(si, ai, ri, si+1)i=1.....N}，对每个数据，使用目标网络计算yi = ri + γ * max(Qw(si+1, a)), 最小目标损失:
L = (1/N * sum((yi - Qw(si, ai)) ** 2))，以此更新当前网络Qw

'''

import gym
import torch
import random
from tqdm import tqdm

import rl_utils
import collections
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ReplayBuffer:
	"""
	经验回放池
	"""
	def __init__(self, capacity):
		"""
		初始化经验回放池大小
		"""
		self.buffer = collections.deque(maxlen=capacity)  # 队列先进先出

	def add(self, state, action, reward, next_state, done):
		"""
		添加状态数据(tuple)至经验回放池中
		"""
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		"""
		从经验回放池中采样数据
		"""
		transitions = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = zip(*transitions)  # 采样数据解包
		return np.array(state), action, reward, np.array(next_state), done

	def size(self):
		"""
		经验回放池存储数据数量
		"""
		return len(self.buffer)

class QNet(torch.nn.Module):
	"""
	构建Q网络
	"""
	def __init__(self,state_dim, hidden_dim, action_dim):
		"""
		初始化Q神经网络参数
		"""
		super(QNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传播
		隐藏层使用Relu激活函数(output > 0)
		"""
		x = F.relu(self.fc1(x))
		return self.fc2(x)

class ConvolutionalQNet(torch.nn.Module):
	"""
	创建Q神经网络模型
	"""
	def __init__(self, action_dim, in_channels=4):
		super(ConvolutionalQNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=(8,), stride=(4,))
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(4,), stride=(2,))
		self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3,), stride=(1,))
		self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
		self.head = torch.nn.Linear(512, action_dim)

	def forward(self, x):
		x = x / 255
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc4(x))
		return self.head(x)

class DQN:
	"""
	DQN模型(算法)
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			learning_rate,
			gamma,
			epsilon,
			target_update,
			device):

		self.action_dim = action_dim  # 动作维度
		self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)  # 创建Q网络
		self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)  # 创建目标Q网络
		self.optimizer = torch.optim.RAdam(self.q_net.parameters(), lr=learning_rate)  # 创建优化器
		self.count = 0  # 计数器，记录更新次数
		self.gamma = gamma  # 折扣因子
		self.epsilon = epsilon  # # epsilon-贪婪策略
		self.target_update = target_update  # 目标网络更新频率
		self.device = device  # 训练设备

	def take_action(self, state):
		"""
		更新下一个动作采取
		"""
		# todo: epsilon - 贪婪策略采取动作
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.action_dim)
		# todo: 采取最大为输出动作
		else:
			state = torch.tensor([state], dtype=torch.float).to(self.device)
			action = self.q_net(state).argmax().item()
		return action

	def update(self, transition_dict):
		"""
		更新目标网络权重
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

		q_values = self.q_net(states).gather(1, actions)  # Q值
		max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 计算目标网络中最大Q值
		q_target = rewards + self.gamma * max_next_q_values * (1 - done)  # 计算TD误差
		dqn_loss = torch.mean(F.mse_loss(q_values, q_target))  # 均方误差损失函数

		# PyTorch中默认梯度会累积，此处将梯度置为0
		self.optimizer.zero_grad()
		# 反向传播更新参数
		dqn_loss.backward()
		self.optimizer.step()

		# 更新目标网络
		if self.count % self.target_update == 0:
			self.target_q_net.load_state_dict(self.q_net.state_dict())
		self.count += 1

def train():
	"""
	训练
	"""
	lr = 2e-3
	num_episodes = 500
	hidden_dim = 128
	gamma = 0.98
	epsilon = 0.01
	target_update = 10
	buffer_size = 10000
	minimal_size = 500
	batch_size = 64
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# 搭建游戏环境
	env_name = 'CartPole-v0'
	env = gym.make(env_name, render_mode="human")
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)

	# 经验回放池
	replay_buffer = ReplayBuffer(buffer_size)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

	return_list = []
	for i in range(10):
		with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
			for i_episode in range(int(num_episodes / 10)):
				episode_return = 0
				state = env.reset()[0]
				done = False
				while not done:
					action = agent.take_action(state)
					next_state, reward, done, _, _ = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)
					state = next_state
					episode_return += reward
					# 当buffer数据的数量超过一定值后,才进行Q网络训练
					if replay_buffer.size() > minimal_size:
						b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
						transition_dict = {
							'states': b_s,
							'actions': b_a,
							'next_states': b_ns,
							'rewards': b_r,
							'done': b_d
						}
						agent.update(transition_dict)
				return_list.append(episode_return)
				if (i_episode + 1) % 10 == 0:
					pbar.set_postfix({
						'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
						'return': '%.3f' % np.mean(return_list[-10:])
					})
				pbar.update(1)

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('DQN on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('DQN on {}'.format(env_name))
	plt.show()


if __name__ == '__main__':
	train()
