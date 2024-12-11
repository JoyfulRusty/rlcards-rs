# -*- coding: utf-8 -*-

# todo: SAC算法

'''

'''

import gym
import torch
import random
import rl_utils
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

class PolicyNetContinuous(torch.nn.Module):
	"""
	连续策略网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
		"""
		初始化网络参数
		"""
		super(PolicyNetContinuous, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
		self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
		self.action_bound = action_bound

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		mu = self.fc_mu(x)
		std = F.softplus(self.fc_std(x))
		dist = Normal(mu, std)
		normal_sample = dist.rsample()  # 重参数化采样
		log_prob = dist.log_prob(normal_sample)
		action = torch.tanh(normal_sample)

		# 计算tanh_normal分布的对数概率密度
		log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
		action = action * self.action_bound
		return action, log_prob


class QNetContinuous(torch.nn.Module):
	"""
	Q连续网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(QNetContinuous, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.fc_out = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x, a):
		"""
		前向传递
		"""
		cat = torch.cat([x, a], dim=1)
		x = F.relu(self.fc1(cat))
		x = F.relu(self.fc2(x))
		return self.fc_out(x)


class SACContinuous:
	"""
	处理连续动作的SAC算法
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			action_bound,
			actor_lr,
			critic_lr,
			alpha_lr,
			target_entropy,
			tau,
			gamma,
			device):
		"""
		初始化参数
		"""
		self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
		self.critic_1 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)
		self.critic_2 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)

		self.target_critic_1 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)
		self.target_critic_2 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)

		# 令目标Q网络的初始参数和Q网络一样
		self.target_critic_1.load_state_dict(self.critic_1.state_dict())
		self.target_critic_2.load_state_dict(self.critic_2.state_dict())

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
		self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

		#使用alpha的log值，可用使训练结果比较稳定
		self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
		self.log_alpha.requires_grad = True  # 对alpha求梯度
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

		self.target_entropy = target_entropy  # 目标熵大小
		self.gamma = gamma
		self.tau = tau
		self.device = device

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		action = self.actor(state)[0]
		return [action.item()]

	def calc_target(self, rewards, next_states, done):
		"""
		计算目标Q值
		"""
		next_actions, log_prob = self.actor(next_states)
		entropy = -log_prob
		q1_value = self.target_critic_1(next_states, next_actions)
		q2_value = self.target_critic_2(next_states, next_actions)
		next_value = torch.min(q1_value, q2_value + self.log_alpha.exp() * entropy)
		td_target = rewards + self.gamma * next_value * (1 - done)

		return td_target

	def soft_update(self, net, target_net):
		"""
		软更新
		"""
		for param_target, param in zip(target_net.parameters(), net.parameters()):
			param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

	def update(self, transition_dict):
		"""
		更新目标网络
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		# 更新两个Q网络
		td_target = self.calc_target(rewards, next_states, done)
		critic_1_q_values = self.critic_1(states).gather(1, actions)
		critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
		critic_2_q_values = self.critic_2(states).gather(1, actions)
		critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

		self.critic_1_optimizer.zero_grad()
		critic_1_loss.backward()
		self.critic_1_optimizer.step()
		self.critic_2_optimizer.zero_grad()
		critic_2_loss.backward()
		self.critic_2_optimizer.step()

		# 更新策略网络
		actor_prob = self.actor(states)
		logg_prob = torch.log(actor_prob + 1e-8)

		# 直接根据概率计算熵
		entropy = -torch.sum(actor_prob * logg_prob, dim=1, keepdim=True)
		q1_value = self.critic_1(states)
		q2_value = self.critic_2(states)

		# 直接根据概率计算期望
		min_q_value = torch.sum(actor_prob * torch.min(q1_value, q2_value), dim=1, keepdim=True)
		actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_q_value)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# 更新alpha值
		alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
		self.log_alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

		# 更新目标网络
		self.soft_update(self.critic_1, self.target_critic_1)
		self.soft_update(self.critic_2, self.target_critic_2)

def train_sac_0():
	env_name = 'Pendulum-v0'
	env = gym.make(env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_bound = env.action_space.high[0]  # 动作最大值
	random.seed(0)
	np.random.seed(0)
	env.seed(0)
	torch.manual_seed(0)
	actor_lr = 3e-4
	critic_lr = 3e-3
	alpha_lr = 3e-4
	num_episodes = 100
	hidden_dim = 128
	gamma = 0.99
	tau = 0.005  # 软更新参数
	buffer_size = 100000
	minimal_size = 1000
	batch_size = 64
	target_entropy = -env.action_space.shape[0]
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
		"cpu")

	replay_buffer = rl_utils.ReplayBuffer(buffer_size)
	agent = SACContinuous(
		state_dim,
		hidden_dim,
		action_dim,
		action_bound,
		actor_lr,
		critic_lr,
		alpha_lr,
		target_entropy,
		tau,
		gamma,
		device)

	return_list = rl_utils.train_off_policy_agent(
		env,
		agent,
		num_episodes,
		replay_buffer,
		minimal_size,
		batch_size)

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('SAC on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('SAC on {}'.format(env_name))
	plt.show()


class PolicyNet(torch.nn.Module):
	"""
	策略网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化策略网络参数
		"""
		super().__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
	"""
	只存在一层隐藏层Q网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化网络参数
		"""
		super().__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return self.fc2(x)


class SAC:
	"""
	SAC
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			actor_lr,
			critic_lr,
			alpha_lr,
			target_entropy,
			tau,
			gamma,
			device):
		"""
		初始化参数
		"""
		# 策略网络
		self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
		# QNet-1
		self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
		# QNet-2
		self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

		# 目标网络
		self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
		self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

		# 令目标Q网络的初始参数和Q网络一样
		self.target_critic_1.load_state_dict(self.critic_1.state_dict())
		self.target_critic_2.load_state_dict(self.critic_2.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
		self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

		# 使用alpha的log值，可使训练结果比较稳定
		self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
		self.log_alpha.requires_grad = True  # 对alpha求梯度
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

		# 目标熵大小
		self.target_entropy = target_entropy
		self.gamma = gamma
		self.tau = tau
		self.device = device

