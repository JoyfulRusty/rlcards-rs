# -*- coding: utf-8 -*-

# todo: Actor-Critic

'''

'''

import gym
import torch
import rl_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
	"""
	构建策略网络
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
	构建Value模型
	"""
	def __init__(self, state_dim, hidden_dim):
		"""
		初始化网络层参数
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


class ActorCritic:
	"""
	构建Actor-Critic模型
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			actor_lr,
			critic_lr,
			gamma,
			device):

		# 策略和价值网络
		self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
		self.critic = ValueNet(state_dim, hidden_dim).to(device)

		# 优化器
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.gamma = gamma
		self.device = device

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		# 策略网络预测动作分布概率
		action_prob = self.actor(state)
		action_dist = torch.distributions.Categorical(action_prob)
		action = action_dist.sample()
		return action.item()

	def update(self, transition_dict):
		"""
		更新价值网络参数
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		# 时序差分目标，计算间距误差
		td_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
		# 计算时序差分误差
		td_delta = td_target - self.critic(states)
		log_prob = torch.log(self.actor(states).gather(1, actions))

		# 计算策略网络的误差值
		actor_loss = torch.mean(-log_prob * td_delta.detach())
		# 计算价值网络的误差损失(均方误差损失函数)
		critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

		# 梯度归零
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()

		# 计算梯度
		actor_loss.backward()
		critic_loss.backward()

		# 更新网络参数
		self.actor_optimizer.step()
		self.critic_optimizer.step()

def trainAC():
	"""
	训练AC模型
	"""
	actor_lr = 1e-3
	critic_lr = 1e-2
	num_episodes = 1000
	hidden_dim = 128
	gamma = 0.98
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# 构建环境
	env_name = "CartPole-v1"
	env = gym.make(env_name, render_mode="human")
	torch.manual_seed(0)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
	return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('Actor-Critic on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('Actor-Critic on {}'.format(env_name))
	plt.show()


if __name__ == '__main__':
	trainAC()