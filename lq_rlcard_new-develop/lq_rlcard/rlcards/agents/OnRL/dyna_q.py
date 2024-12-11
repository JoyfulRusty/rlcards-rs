# -*- coding: utf-8 -*-

# todo: Dyna-Q 使用一种叫做 Q-planning 的方法来基于模型生成一些模拟数据，然后用模拟数据和真实数据一起改进策略。

'''
Dyna-Q使用一种叫做Q-planning的方式来基于模型生成一些模型数据，然后使用模拟数据和真实数据一起改进策略，Q-planning每次选取一个曾经访问过的状态s,
采取一个曾经在该状态下执行过的动作a，通过模型得到转移后的状态s`以及奖励r，并根据这个模拟数据(s, a, r, s`)，用Q-learning的更新方式来更新动作价值函数

用£-greedy策略根据Q选择当前状态s下的动作a得到环境返回的r, s`:
Q(s, a) = Q(s, a) + α[r + γ * maxQ(s`, a`) - Q(s, a)]
M(s, a) = r, s`

随机选择一个曾经访问过的状态sm，采取一个曾经在状态sm，下执行过的动作am:
rm, s`m = M(sm, am)
Q(sm, am) = Q(sm, am) + α[r + γ * maxQ(s`, a`) - Q(s, a)]

在每次与环境进行交互执行一次Q-learning之后，Dyna-Q会做N次Q-planning，其中Q-planning的次数N是一个实现可选择的超参数，当其为0时就是普通的Q-learning
注意: 上述Dyna-Q算法是执行在一个离散并且确定的环境中，所有当看到一条经验数据(s, a, r, s`)时，可直接对模型做出更新，即M(s, a) = r, s`
'''

import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class CliffWalkingEnv:
	"""
	悬崖漫步环境
	"""
	def __init__(self, n_col, n_row):
		"""
		初始化参数
		"""
		self.n_col = n_col
		self.n_row = n_row
		self.x = 0  # 记录当前智能体横坐标
		self.y = self.n_row - 1  # 记录当前智能体位置的纵坐标

	def step(self, action):
		"""
		更新当前位置
		4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
		"""
		change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
		self.x = min(self.n_col - 1, max(0, self.x + change[action][0]))
		self.y = min(self.n_row - 1, max(0, self.y + change[action][1]))

		# 下一个状态、奖励，是否达到
		next_state = self.y * self.n_col + self.x
		reward = -1
		done = False

		# 下一个位置在悬崖或者目标
		if self.y == self.n_row - 1 and self.x > 0:
			done = True
			if self.x != self.n_col - 1:
				reward = -100

		return next_state, reward, done

	def reset(self):
		"""
		回归初始化状态，坐标轴远点在左上角
		"""
		self.x = 0
		self.y = self.n_row - 1
		return self.y * self.n_col + self.x

class DynaQ:
	"""
	Dyna-Q算法
	"""
	def __init__(self, n_col, n_row, epsilon, alpha, gamma, n_planning, n_action=4):
		"""
		初始化参数:
		"""
		self.q_table = np.zeros([n_row * n_col, n_action])  # 初始化Q(s, a)表格
		self.n_action = n_action  # 动作个数
		self.alpha = alpha  # 学习率
		self.gamma = gamma  # 折扣因子
		self.model = {}  # 环境模型
		self.epsilon = epsilon  # epsilon-贪婪策略中的参数
		self.n_planning = n_planning  # 执行Q-planning的次数，对应1次Q-learning

	def take_action(self, state):
		"""
		选取动作
		"""
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.n_action)
		else:
			action = np.argmax(self.q_table[state])
		return action

	def q_learning(self, s0, a0, r, s1):
		"""
		q-learning算法更新公式
		计算差距，更新Q表
		"""
		max_q = self.q_table[s1].max()
		# print()
		# print("最大Q值: ", max_q)
		td_error = r + self.gamma * self.q_table[s1].max() - self.q_table[s0, a0]
		self.q_table[s0, a0] += self.alpha * td_error

	def update_q_value(self, s0, a0, r, s1):
		"""
		更新Q值
		"""
		self.q_learning(s0, a0, r, s1)
		# 添加状态数据到模型中
		self.model[(s0, a0)] = r, s1
		# Q-planning循环更新
		for _ in range(self.n_planning):
			# 随机选择曾经遇到过的状态动作
			(s, a), (r, s_) = random.choice(list(self.model.items()))
			self.q_learning(s, a, r, s_)

def dynaQCliffWalking(n_planning):
	"""
	Dyna Q 悬崖漫步环境
	"""
	n_col = 12
	n_row = 4
	env = CliffWalkingEnv(n_col, n_row)
	epsilon = 0.01
	alpha = 0.1
	gamma = 0.9

	agent = DynaQ(n_col, n_row, epsilon, alpha, gamma, n_planning)
	num_episodes = 50  # 智能体在环境中运行多少条序列

	# 记录每一条序列的回报
	return_list= []
	# 显示10个进度条
	for i in range(100):
		with tqdm(total=int(num_episodes / 10), desc='Iteration % d' % i) as bar:
			# 每个进度条的序列
			for i_episode in range(int(num_episodes / 10)):
				episode_return = 0
				state = env.reset()
				done = False
				while not done:
					action = agent.take_action(state)
					next_state, reward, done = env.step(action)
					# 回报计算，不再进行折扣因子衰减
					episode_return += reward
					agent.update_q_value(state, action, reward, next_state)
					state = next_state
				return_list.append(episode_return)
				# 每10条序列打印一下这10条序列的平均回报
				if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
					bar.set_postfix({
						'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
						'return': '%.3f' % np.mean(return_list[-10:])
					})
				bar.update(1)
	return return_list


def train_dyna_q():
	"""
	训练dyna
	"""
	np.random.seed(0)
	random.seed(0)
	n_planning_list = [0, 2, 20]
	for n_planning in n_planning_list:
		print('Q-planning步数为：%d' % n_planning)
		time.sleep(0.5)
		return_list = dynaQCliffWalking(n_planning)
		episodes_list = list(range(len(return_list)))
		plt.plot(episodes_list,return_list, label=str(n_planning) + ' planning steps')
	plt.legend()
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('Dyna-Q on {}'.format('Cliff Walking'))
	plt.show()


if __name__ == '__main__':
	train_dyna_q()