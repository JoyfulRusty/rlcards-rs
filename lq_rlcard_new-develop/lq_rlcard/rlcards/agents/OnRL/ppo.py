# -*- coding: utf-8 -*-

# todo: ppo算法

import gym
import torch
import rl_utils

import torch.nn.functional as F
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
	"""
	策略网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化策略网络参数
		"""
		super(PolicyNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
	"""
	价值网络
	"""
	def __init__(self, state_dim, hidden_dim):
		"""
		初始化参数
		"""
		super(ValueNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return self.fc2(x)


class PPO:
	"""
	PPO算法，采用截断方式
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			actor_lr,
			critic_lr,
			lm_bda,
			epochs,
			eps,
			gamma,
			device):
		"""
		初始化PPO参数
		"""
		# 构建两个模型(策略网络和价值网络)
		self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
		self.critic = ValueNet(state_dim, hidden_dim).to(device)

		# 分别构建两个优化器
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.gamma = gamma
		self.lm_bda = lm_bda
		self.epochs = epochs  # 一条序列的数据，用来训练轮数
		self.eps = eps   # PPO中截断范围的参数
		self.device = device

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		action_prob = self.actor(state)
		action_dist = torch.distributions.Categorical(action_prob)
		action = action_dist.sample()
		return action.item()

	def update(self, transition_dist):
		"""
		更新网络
		"""
		states = torch.tensor(transition_dist['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dist['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dist['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dist['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dist['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		td_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
		td_delta = td_target - self.critic(states)
		advantage = rl_utils.compute_advantage(self.gamma, self.lm_bda, td_delta.cpu()).to(self.device)
		old_log_prob = torch.log(self.actor(states).gather(1, actions)).detach()

		for _ in range(self.epochs):
			log_prob = torch.log(self.actor(states).gather(1, actions))
			ratio = torch.exp(log_prob - old_log_prob)
			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
			actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
			critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

			# 梯度归零
			self.actor_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()

			# 反向传播
			actor_loss.backward()
			critic_loss.backward()

			# 更新
			self.actor_optimizer.step()
			self.critic_optimizer.step()

def train_ppo():
	"""
	ppo训练
	"""
	actor_lr = 1e-3
	critic_lr = 1e-3
	num_episodes = 500
	hidden_dim = 128
	gamma = 0.98
	lm_bda = 0.95
	epochs = 10
	eps = 0.2
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	env_name = 'CartPole-v1'
	env = gym.make(env_name, render_mode='human')
	env.seed(0)
	torch.manual_seed(0)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lm_bda,epochs, eps, gamma, device)
	return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('PPO on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('PPO on {}'.format(env_name))
	plt.show()


class PolicyNetContinuous(torch.nn.Module):
	"""
	连续策略
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化参数
		"""
		super(PolicyNetContinuous, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
		self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		mu = 2.0 * torch.tanh(self.fc_mu(x))
		std = F.softplus(self.fc_std(x))
		return mu, std


class PPOContinuous:
	"""
	处理连续动作的PPO算法
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			actor_lr,
			critic_lr,
			lm_bda,
			epochs,
			eps,
			gamma,
			device):
		"""
		初始化参数
		"""
		self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,).to(device)
		self.critic = ValueNet(state_dim, hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.gamma = gamma
		self.lm_bda = lm_bda
		self.epochs = epochs
		self.eps = eps
		self.device = device

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		mu, sigma = self.actor(state)
		action_dist = torch.distributions.Normal(mu, sigma)
		action = action_dist.sample()
		return [action.item()]

	def update(self, transition_dict):
		"""
		更新模型
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'],  dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		# 与T R P O一样，对奖励进行修改，方便训练
		rewards = (rewards + 8.0) / 8.0
		td_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
		td_delta = td_target - self.critic(states)
		advantage = rl_utils.compute_advantage(self.gamma, self.lm_bda,td_delta.cpu()).to(self.device)
		mu, std = self.actor(states)
		action_dists = torch.distributions.Normal(mu.detach(), std.detach())
		# 动作是正态分布
		old_log_prob = action_dists.log_prob(actions)

		for _ in range(self.epochs):
			mu, std = self.actor(states)
			action_dists = torch.distributions.Normal(mu, std)
			log_prob = action_dists.log_prob(actions)
			ratio = torch.exp(log_prob - old_log_prob)
			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
			actor_loss = torch.mean(-torch.min(surr1, surr2))
			critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
			self.actor_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()
			actor_loss.backward()
			critic_loss.backward()
			self.actor_optimizer.step()
			self.critic_optimizer.step()

def train_ppo_continuous():
	actor_lr = 1e-4
	critic_lr = 5e-3
	num_episodes = 2000
	hidden_dim = 128
	gamma = 0.9
	lm_bda = 0.9
	epochs = 10
	eps = 0.2
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	env_name = 'Pendulum-v1'
	env = gym.make(env_name, render_mode='human')
	env.seed(0)
	torch.manual_seed(0)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]  # 连续动作空间
	agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lm_bda, epochs, eps, gamma, device)
	return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('PPO on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 21)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('PPO on {}'.format(env_name))
	plt.show()


if __name__ == '__main__':
	train_ppo_continuous()