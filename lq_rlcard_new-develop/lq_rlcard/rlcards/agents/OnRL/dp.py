# -*- coding: utf-8 -*-

import copy

# todo: 动态规划(dynamic programming)
'''
1.策略迭代(policy iteration): 策略评估(policy evaluation)和策略提升(policy improvement)
2.价值迭代(value iteration)
3.贝尔曼期望方程:
	VΠ(s) = 累加求和: Π(a|s) * (r(s, a) + γ * 累加求和: p(s`|s, a)VΠ(s`))
	策略概率: Π(a|s), 策略Π在状态s下采取动作a的概率
	奖励函数: r(s, a)，奖励可以同时取决于状态s和动作a，在将i函数只取决于s时，则退化为r(s)
	状态转移函数: P(s`|s, a)，表示在状态s执行动作a之后达到状态s`的概率
	
4.策略提升定理: 可直接贪心的在每一个状态选择动作价值最大的动作，公式如下:
	Π`(s) = argmax(QΠ(s, a)) = argmax{r(s, a) + γ * (累加求和： P(s`|s, a)VΠ(s`))}
	
5.策略迭代算法的过程如下：对当前的策略进行策略评估，得到其状态价值函数，然后根据该状态价值函数进行策略提升以得到一个更好的新策略，
  接着继续评估新策略、提升策略……直至最后收敛到最优策略
  
6.价值迭代算法: 价值迭代可看成一种动态规划过程，其利用了贝尔曼最优方程:
	V^*(s) = max{r(s, a) + γ * (累加求和: P(s`|s, a)V^*(s`))}
	迭代更新公式:
		V^(k+1)(s) = max{r(s, a) + γ * (累加求和: P(s`|s, a)V^K(s`))}
'''


class CliffWalkingEnv:
	"""
	悬崖漫步环境
	"""
	def __init__(self, n_col=12, n_row=4):
		"""
		初始化网格
		"""
		self.n_col = n_col  # 网格列
		self.n_row = n_row  # 网格行
		# 转移矩阵p[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
		self.p = self.create_policy()

	def create_policy(self):
		"""
		初始化策略p
		4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
		"""
		P = [[[] for _ in range(4)] for _ in range(self.n_row * self.n_col)]
		# 定义左上角
		change = [[0,-1], [0, 1], [-1, 0], [1, 0]]
		for i in range(self.n_row):
			for j in range(self.n_col):
				for a in range(4):
					# 位置在悬崖或者目标状态，因为无法继续交互，任何动作奖励都为0
					if i == self.n_row - 1 and j > 0:
						P[i * self.n_col + j][a] = [(1, i * self.n_col + j, 0, True)]
						continue
					# 其他位置
					next_x = min(self.n_col - 1, max(0, j + change[a][0]))
					next_y = min(self.n_row - 1, max(0, i + change[a][1]))
					next_state = next_y * self.n_col + next_x
					reward = -1
					done = False
					# 下一个位置在选择或者终点
					if next_y == self.n_row and next_x > 0:
						done = True
						# 下一个位置在悬崖
						if next_x != self.n_col - 1:
							reward = -100
					P[i * self.n_col + j][a] = [(1, next_state, reward, done)]
		return P


class PolicyIteration:
	"""
	策略迭代算法
	"""
	def __init__(self, env, theta, gamma):
		"""
		初始化策略迭代算法参数
		"""
		self.env = env
		self.v = [0] * self.env.n_col * self.env.n_row  # 初始化价值为0
		self.pi = [
			[0.25, 0.25, 0.25, 0.25]
			for _ in range(self.env.n_col * self.env.n_row)
		]  # 初始化为均匀随机策略
		self.theta = theta  # 策略评估收敛阈值
		self.gamma = gamma  # 折扣因子

	def policy_evaluation(self):
		"""
		策略评估
		"""
		cnt = 1  # 计数器
		while 1:
			max_diff = 0
			new_v = [0] * self.env.n_col * self.env.n_row
			for s in range(self.env.n_col * self.env.n_row):
				# 开始计算状态s下的所有Q(s, a)价值
				qsa_list = []
				for a in range(4):
					qsa = 0
					for res in self.env.p[s][a]:
						p, next_state, r, done = res
						# 算法更新公式
						qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
					# 环境比较特殊, 奖励和下一个状态有关, 所以需要和状态转移概率相乘
					qsa_list.append(self.pi[s][a] * qsa)
				new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
				max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
			self.v = new_v
			if max_diff < self.theta:
				break  # 满足收敛条件,退出评估迭代
			cnt += 1
		print("策略评估进行%d轮后完成" % cnt)

	def policy_improvement(self):
		"""
		策略提升
		"""
		for s in range(self.env.n_row * self.env.n_col):
			qsa_list = []
			for a in range(4):
				qsa = 0
				for res in self.env.p[s][a]:
					p, next_state, r, done = res
					# 算法更新公式
					qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
				qsa_list.append(qsa)
			max_q = max(qsa_list)
			cnt_q = qsa_list.count(max_q)  # 计算有几个动作得到了最大的Q值
			# 动作均匀分布概率
			self.pi[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_list]
		print("策略提升完成")
		return self.pi

	def policy_iteration(self):
		"""
		策略迭代
		"""
		while 1:
			self.policy_evaluation()
			old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝，方便进行比较
			new_pi = self.policy_improvement()
			if old_pi == new_pi: break

def print_agent(agent, action_meaning, disaster, end):
	"""
	输出代理新
	"""
	print("状态价值：")
	for i in range(agent.env.n_row):
		for j in range(agent.env.n_col):
			# 为了输出美观,保持输出6个字符
			print('%6.6s' % ('%.3f' % agent.v[i * agent.env.n_col + j]), end=' ')
		print()

	print("策略：")
	for i in range(agent.env.n_row):
		for j in range(agent.env.n_col):
			# 一些特殊的状态,例如悬崖漫步中的悬崖
			if (i * agent.env.n_col + j) in disaster:
				print('****', end=' ')
			elif (i * agent.env.n_col + j) in end:  # 目标状态
				print('EEEE', end=' ')
			else:
				a = agent.pi[i * agent.env.n_col + j]
				pi_str = ''
				for k in range(len(action_meaning)):
					pi_str += action_meaning[k] if a[k] > 0 else 'o'
				print(pi_str, end=' ')
		print()


env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

'''
策略评估进行49轮后完成
策略提升完成
策略评估进行5轮后完成
策略提升完成
策略评估进行1轮后完成
策略提升完成
状态价值：
-3.439 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 
-2.710 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 
-1.900 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 
-1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
策略：
ovo> ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo 
ovo> ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo 
ovo> ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo 
ooo> **** **** **** **** **** **** **** **** **** **** EEEE 
'''

class ValueIteration:
	"""
	价值迭代算法
	"""
	def __init__(self, env, theta, gamma):
		"""
		初始化价值迭代算法参数
		"""
		self.env = env
		self.v = [0] * self.env.n_col * self.env.n_row  # 初始化价值为0
		self.theta = theta  # 价值收敛阈值
		self.gamma = gamma
		# 价值迭代结束后得到的策略
		self.pi = [None for i in range(self.env.n_col * self.env.n_row)]

	def value_iteration(self):
		"""
		价值迭代过程
		"""
		cnt = 0
		while 1:
			max_diff = 0
			new_v = [0] * self.env.n_col * self.env.n_row
			for s in range(self.env.n_col * self.env.n_row):
				qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
				for a in range(4):
					qsa = 0
					for res in self.env.p[s][a]:
						p, next_state, r, done = res
						qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
					qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别
				new_v[s] = max(qsa_list)
				max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
			self.v = new_v
			if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
			cnt += 1
		print("价值迭代一共进行%d轮" % cnt)
		self.get_policy()

	def get_policy(self):
		"""
		根据价值函数导出一个贪婪策略
		"""
		for s in range(self.env.n_row * self.env.n_col):
			qsa_list = []
			for a in range(4):
				qsa = 0
				for res in self.env.p[s][a]:
					p, next_state, r, done = res
					qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
				qsa_list.append(qsa)
			max_q = max(qsa_list)
			cnt_q = qsa_list.count(max_q)  # 计算有几个动作得到了最大的Q值
			# 动作均分概率
			self.pi[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_list]

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

'''
价值迭代一共进行4轮
状态价值：
-3.439 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 -2.710 
-2.710 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 -1.900 
-1.900 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 -1.000 
-1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
策略：
ovo> ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo 
ovo> ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo 
ovo> ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo ovoo 
ooo> **** **** **** **** **** **** **** **** **** **** EEEE 
'''


import gym
env = gym.make("FrozenLake-v1", render_mode="human")  # 创建环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
# env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
	for a in env.P[s]:
		for s_ in env.P[s][a]:
			if s_[2] == 1.0:  # 获得奖励为1,代表是目标
				ends.add(s_[1])
			if s_[3] == True:
				holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
	print(env.P[14][a])

'''
SFFF
FHFH
FFFH
HFFG
冰洞的索引: {11, 12, 5, 7}
目标的索引: {15}
[(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False),
 (0.3333333333333333, 14, 0.0, False)]
[(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False),
 (0.3333333333333333, 15, 1.0, True)]
[(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True),
 (0.3333333333333333, 10, 0.0, False)]
[(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False),
 (0.3333333333333333, 13, 0.0, False)]
'''

# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

'''
策略评估进行25轮后完成
策略提升完成
策略评估进行58轮后完成
策略提升完成
状态价值：
 0.069  0.061  0.074  0.056
 0.092  0.000  0.112  0.000
 0.145  0.247  0.300  0.000
 0.000  0.380  0.639  0.000
策略：
<ooo ooo^ <ooo ooo^
<ooo **** <o>o ****
ooo^ ovoo <ooo ****
**** oo>o ovoo EEEE
'''

action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

'''
价值迭代一共进行60轮
状态价值：
 0.069  0.061  0.074  0.056
 0.092  0.000  0.112  0.000
 0.145  0.247  0.300  0.000
 0.000  0.380  0.639  0.000
策略：
<ooo ooo^ <ooo ooo^
<ooo **** <o>o ****
ooo^ ovoo <ooo ****
**** oo>o ovoo EEEE
'''