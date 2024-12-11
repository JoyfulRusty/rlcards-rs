# -*- coding: utf-8 -*-

# todo: d d p g 算法

'''

'''

import gym
import torch
import random
import rl_utils
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

class PolicyNet(torch.nn.Module):
	"""
	策略网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
		"""
		初始化网络参数
		"""
		super(PolicyNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
		# 环境可接受的动作最大值
		self.action_bound = action_bound

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return torch.tanh(self.fc2(x) * self.action_bound)

class QNet(torch.nn.Module):
	"""
	Q值网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化参数
		"""
		super(QNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim+action_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.fc_out = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x, a):
		"""
		前向传递
		"""
		# 拼接状态和动作
		x = torch.cat([x, a], dim=1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc_out(x)


class DDPG:
	"""
	确定性策略 D D P G 算法
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			action_bound,
			sigma,
			actor_lr,
			critic_lr,
			tau,
			gamma,
			device):
		"""
		初始化参数
		"""
		# 训练网络
		self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
		self.critic = QNet(state_dim, hidden_dim, action_dim).to(device)
		# 目标更新网络
		self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
		self.target_critic = QNet(state_dim, hidden_dim, action_dim).to(device)
		# 初始化目标价值网卡并设置和策略网络相同的参数
		self.target_actor.load_state_dict(self.actor.state_dict())
		# 初始化目标价值网卡并设置和价值网络相同的参数
		self.target_critic.load_state_dict(self.critic.state_dict())
		# 优化器
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.gamma = gamma
		self.sigma = sigma  # 高斯噪声标准化，均值直接设为0
		self.tau = tau  # 目标网络软更新参数
		self.action_dim = action_dim
		self.device = device

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		# 策略网络预测动作
		action = self.actor(state).item()
		# 添加噪声，增加探索
		action = action + self.sigma * np.random.randn(self.action_dim)
		return action

	def soft_update(self, net, target_net):
		"""
		参数软更新
		"""
		for param_target, param in zip(target_net.parameters(), net.parameters()):
			param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

	def update(self, transition_dict):
		"""
		更新模型参数
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		# 下一个Q值
		next_q_values = self.target_critic(next_states, self.target_actor(next_states))
		q_targets = rewards + self.gamma * next_q_values * (1 - done)
		critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		actor_loss = -torch.mean(self.critic(states, self.actor(states)))
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()


def train_pg():
	"""
	训练 D D P G 模型
	"""
	actor_lr = 3e-4
	critic_lr = 3e-3
	num_episodes = 200
	hidden_dim = 64
	gamma = 0.98
	tau = 0.005  # 软更新参数
	buffer_size = 10000
	minimal_size = 1000
	batch_size = 64
	sigma = 0.01  # 高斯噪声标准差
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# 构建环境
	env_name = 'Pendulum-v1'
	env = gym.make(env_name, render_mode='human')
	random.seed(0)
	np.random.seed(0)
	env.seed(0)  # 0.25.2 gym
	torch.manual_seed(0)
	replay_buffer = rl_utils.ReplayBuffer(buffer_size)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_bound = env.action_space.high[0]  # 动作最大值
	agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
	return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('DDPG on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('DDPG on {}'.format(env_name))
	plt.show()


if __name__ == '__main__':
	train_pg()