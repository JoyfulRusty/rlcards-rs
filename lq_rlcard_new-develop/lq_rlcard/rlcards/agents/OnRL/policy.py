# -*- coding: utf-8 -*-

# todo: 策略梯度算法

'''

'''

import gym
import torch
import rl_utils
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm


class PolicyNet(torch.nn.Module):
	"""
	构建策略梯度网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(PolicyNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传播
		"""
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=1)


class Reinforce:
	"""
	构建加强策略模型
	"""
	def __init__(
			self,
			state_dim,
			hidden_dim,
			action_dim,
			learning_rate,
			gamma,
			device):
		self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # Adam优化器
		self.gamma = gamma  # 折扣因子
		self.device = device

	def take_action(self, state):
		"""
		根据动作概率分布随机采样
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		action_prob = self.policy_net(state)
		action_dist = torch.distributions.Categorical(action_prob)

		# 动作采样
		action = action_dist.sample()

		return action.item()

	def update(self, transition_dict):
		"""
		更新网络
		"""
		reward_list = transition_dict['rewards']
		state_list = transition_dict['states']
		action_list = transition_dict['actions']

		G = 0  # 总的期望回报
		self.optimizer.zero_grad()
		# 从后玩前进行计算
		for i in reversed(range(len(reward_list))):
			reward = reward_list[i]
			state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
			action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
			log_prob = torch.log(self.policy_net(state).gather(1, action))

			# 计算总的期望回报
			G = self.gamma * G + reward
			# 计算损失函数
			loss = -log_prob * G
			# 反向传播计算梯度
			loss.backward()
		# 梯度下降
		self.optimizer.step()

def train_policyNet():
	"""
	训练策略模型
	"""
	learning_rate = 1e-2
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
	agent = Reinforce(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

	return_list = []
	for i in range(10):
		with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as bar:
			for i_episode in range(int(num_episodes / 10)):
				episode_return = 0
				transition_dict = {
					'states': [],
					'actions': [],
					'next_states': [],
					'rewards': [],
					'done': []
				}
				state = env.reset()
				done = False
				while not done:
					action = agent.take_action(state)
					next_state, reward, done, _ = env.step(action)
					transition_dict['states'].append(state)
					transition_dict['actions'].append(action)
					transition_dict['next_states'].append(next_state)
					transition_dict['rewards'].append(reward)
					transition_dict['done'].append(done)

					# 下一个状态
					state = next_state
					episode_return += reward
				return_list.append(episode_return)
				agent.update(transition_dict)
				# 输出训练日志
				if (i_episode + 1) % 10 == 0:
					bar.set_postfix({
						'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
						'return': '%.3f' % np.mean(return_list[-10:])
					})
				bar.update(1)

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('REINFORCE on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('REINFORCE on {}'.format(env_name))
	plt.show()


if __name__ == '__main__':
	train_policyNet()