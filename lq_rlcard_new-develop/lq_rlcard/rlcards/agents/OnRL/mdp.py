# -*- coding: utf-8 -*-

# todo: 马尔可夫决策过程
'''
1.马尔可夫性质: 当且仅当某时刻的状态只取决于上一时刻的状态时，这样的一个随机过程被称为具有马尔可夫性质
2.公式表示： P(St+1|St) = P(St+1|S1......,St)，当前状态时未来的充分统计量，即下一个状态只取决于当前状态，而不会收到过去状态的影响
3.马尔可夫链: 具有马尔可夫性质的随机过程，通常使用元素(S, P)描述一个马尔可夫过程，S为有限数量的状态集合，P为状态转移矩阵
   3.1 矩阵P中第i行第j列元素P(sj|si)=P(St+1=sj|St=si)，表示状态si转移到sj的概率，称P(s`|s)为状态转移函数，
   3.2 从某个状态出发，达到其他状态的概率和必须为1，即状态转移矩阵P的每一行的和为1
4.回报Gt=Rt + γRt+1 + γ^2Rt+2 +....γ^kRt+k
5.价值函数: V(s)=E[Gt|St=s] -> V(s)=E[Rt|St=s]=r(s)
6.状态价值函数: VΠ(s) = EΠ[Gt|St=s] -> 基于策略Π的状态价值函数，定位为从s出发遵循策略Π能获得的期望回报
7.动作价值函数: QΠ(s, a) = EΠ[Gt|St=s, At=a] -> 表示基于MDP遵循策略Π时，对当前状态s执行动作a得到的期望回报
   7.1 状态价值函数和动作价值函数之间的关系：
	   在使用策略Π中，状态s的价值等于在该状态下基于策略Π采取所有动作的概率与响应的价值相乘，再求和的结果:
		公式: QΠ(s, a) = r(s, a) + r * sum(P(s`|s, a))VΠ(s`)
8.使用策略Π时，状态s下采取动作a的降至等于即时奖励加上经过衰减后的所有可能的下一个状态的状态转移概率与响应的价值的乘积:
	公式: QΠ(s, a) = r(s, a) + γ * 累加求和: P(s`|s, a)VΠ(s`)

9.蒙特卡洛公式:
	9.1 计算从这个状态出发的回报再求其期望: VΠ(s) = EΠ[Gt|St=s]≈1/N * 累加求和: Gt(i).N
	9.2 对每一条序列中的每一步时间步t的状态s进行以下操作:
		更新状态s的计数器: N(s) = N(s) + 1
		更新状态s的总回报: M(s) = M(s) + Gt
	9.3 每一个状态的价值被估计为回报的平均值: V(s)=M(s)/N(s)
		根据大数定律，当N(s)趋于∞时，有V(s) -> VΠ(s)，计算回报的期望时，除了可以把所有的回报加起来除以次数
		另一种增量更新的方法，对于每个状态s和对应的回报G，进行如下计算:
		N(s) = N(s) + 1
		V(s) = V(s) + [1/N(s) * (G - V(S)]

'''

import numpy as np

# 随机数
np.random.seed(0)

# 定义状态转移概率矩阵P
P = [
	[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
	[0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
	[0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
	[0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

p = np.array(P)

# 定义奖励函数
rewards = [-1, -2, -2, 10, 1, 0]

# 定义折扣因子
gamma = 0.5

# 状态序列: s1 -> s2 -> s3 -> s6
chain = [1, 2, 3, 6]
start_index = 0

def compute_return(start_index, chain, gamma):
	"""
	给定一条序列
	计算从某个索引(起始状态)开始到序列最后(终止状态)得到的回报
	"""
	G = 0
	for i in reversed(range(start_index, len(chain))):
		G = gamma * G + rewards[chain[i] - 1]
	return G

G = compute_return(start_index, chain, gamma)
print("根据本序列计算获取到的回报奖励: %s" % G)

def compute_mrp_nums(p, rewards, gamma, states_num):
	"""
	利用贝尔曼方程的矩阵形式计算解析, states_num为MRP状态数
	"""
	# 写成列向量的形式
	rewards = np.array(rewards).reshape((-1, 1))
	value = np.dot(
		np.linalg.inv(np.eye(states_num, states_num) - gamma * p),
		rewards
	)

	return value

# 马尔可夫奖励过程(MRP)
# 马尔可夫决策过程(MDP)

V = compute_mrp_nums(p, rewards, gamma, 6)
print("MRP中每个状态价值分别为: \n", V)

'''
MRP中每个状态价值分别为: 
 [[-2.01950168]
 [-2.21451846]
 [ 1.16142785]
 [10.53809283]
 [ 3.58728554]
 [ 0.        ]]
'''

# ===================#@====#@===================
#   ================= MDP ===================

S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合

# 状态转移函数
P = {
	"s1-保持s1-s1": 1.0,
	"s1-前往s2-s2": 1.0,
	"s2-前往s1-s1": 1.0,
	"s2-前往s3-s3": 1.0,
	"s3-前往s4-s4": 1.0,
	"s3-前往s5-s5": 1.0,
	"s4-前往s5-s5": 1.0,
	"s4-概率前往-s2": 0.2,
	"s4-概率前往-s3": 0.4,
	"s4-概率前往-s4": 0.4,
}

# 奖励函数
R = {
	"s1-保持s1": -1,
	"s1-前往s2": 0,
	"s2-前往s1": -1,
	"s2-前往s3": -2,
	"s3-前往s4": -2,
	"s3-前往s5": 0,
	"s4-前往s5": 10,
	"s4-概率前往": 1,
}

gamma_0 = 0.5  # 折扣因子

# 马尔可夫决策过程(MDP)
MDP = (S, A, P, R, gamma_0)

# 策略1: 随机策略
Pi_1 = {
	"s1-保持s1": 0.5,
	"s1-前往s2": 0.5,
	"s2-前往s1": 0.5,
	"s2-前往s3": 0.5,
	"s3-前往s4": 0.5,
	"s3-前往s5": 0.5,
	"s4-前往s5": 0.5,
	"s4-概率前往": 0.5,
}

# 策略2
Pi_2 = {
	"s1-保持s1": 0.6,
	"s1-前往s2": 0.4,
	"s2-前往s1": 0.3,
	"s2-前往s3": 0.7,
	"s3-前往s4": 0.5,
	"s3-前往s5": 0.5,
	"s4-前往s5": 0.1,
	"s4-概率前往": 0.9,
}

def join(str1, str2):
	"""
	将输入的两个字符通过"-"连接
	便于使用上述定义的P, R变量
	"""
	return str1 + '-' + str2

gamma_1 = 0.5

# 转换后的MRP状态转移矩阵
P_from_mdp_to_mrp = [
	[0.5, 0.5, 0.0, 0.0, 0.0],
	[0.5, 0.0, 0.5, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.5, 0.5],
	[0.0, 0.1, 0.2, 0.2, 0.5],
	[0.0, 0.0, 0.0, 0.0, 1.0],
]

P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute_mrp_nums(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma_1, 5)
print("MDP中每个状态价值分别为: \n", V)

'''
MDP中每个状态价值分别为: 
 [[-1.22555411]
 [-1.67666232]
 [ 0.51890482]
 [ 6.0756193 ]
 [ 0.        ]]
'''

def sample(mdp, pi, time_step_max, number):
	"""
	采样函数，策略pi，限制最长时间步time_step_max
	总共采样序列数number
	"""
	S, A, P, R, gamma = mdp
	episodes = []
	for _ in range(number):
		episode = []
		time_step = 0
		# 随机选择一个除以s5以外的状态s为起点
		s = S[np.random.randint(4)]
		a = 0
		r = 0
		s_next = 0
		# 当前状态为终止状态或者时间步长太长时，一次采样结束
		while s != 's5' and time_step < time_step_max:
			time_step += 1
			rand, temp = np.random.rand(), 0
			# 在状态s下根据策略选择动作
			for a_opt in A:
				temp += pi.get(join(s, a_opt), 0)
				if temp > rand:
					a = a_opt
					r = R.get(join(s, a), 0)
					break

			rand, temp = np.random.rand(), 0
			# 根据状态转移概率得到下一个状态s_next
			for s_opt in S:
				temp += P.get(join(join(s, a), s_opt), 0)
				if temp > rand:
					s_next = s_opt
					break

			# 把(s,a,r,s_next)元组放入序列中
			episode.append((s, a, r, s_next))
			# s_next变成当前状态,开始接下来的循环
			s = s_next
		episodes.append(episode)
	return episodes

# 采样5次,每个序列最长不超过1000步
episodes = sample(MDP, Pi_1, 20, 5)
print('第一条序列: \n', episodes[0])
print('第二条序列: \n', episodes[1])
print('第五条序列: \n', episodes[4])

'''
第一条序列: 
 [('s1', '前往s2', 0, 's2'), ('s2', '前往s3', -2, 's3'), ('s3', '前往s5', 0, 's5')]
 
第二条序列: 
 [('s4', '概率前往', 1, 's4'), ('s4', '前往s5', 10, 's5')]
 
第五条序列: 
 [('s2', '前往s3', -2, 's3'), ('s3', '前往s4', -2, 's4'), ('s4', '前往s5', 10, 's5')]
'''

def mc_value(episodes, v, n, gamma):
	"""
	对所有采样序列计算所有状态价值
	"""
	for episode in episodes:
		G = 0
		# 一个序列从后往前计算
		for i in range(len(episode) - 1, -1, -1):
			(s, a, r, s_next) = episode[i]
			G = r + gamma * G
			n[s] = n[s] + 1
			v[s] = v[s] + (G - v[s]) / n[s]

time_step_max = 20
# 采样1000次
episodes = sample(MDP, Pi_1, time_step_max, 1000)

gamma_2 = 0.5
v = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
n = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
mc_value(episodes, v, n, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值为: \n", v)

'''
使用蒙特卡洛方法计算MDP的状态价值为: 
 {'s1': -1.2307515763768933, 's2': -1.695303274970171, 's3': 0.4938453462185593, 's4': 6.022334993458098, 's5': 0}
'''

def occupancy(episodes, s, a, time_step_max, gamma):
	"""
	计算状态动作对(s, a)出现的频率，以此来估算策略的占用度量
	"""
	rho = 0
	total_times = np.zeros(time_step_max)  # 记录每个时间步长t各被经历几次
	occur_times = np.zeros(time_step_max)  # 记录(s_t, a_t)=(s, a)次数

	# 循环计算
	for episode in episodes:
		for i in range(len(episode)):
			(s_opt, a_opt, r, s_next) = episode[i]
			total_times[i] += 1
			if s == s_opt and a == a_opt:
				occur_times[i] += 1

	# 计算回报(起始到结束)
	for i in reversed(range(time_step_max)):
		if total_times[i]:
			rho += gamma ** i * occur_times[i] / total_times[i]

	return (1- gamma) * rho

gamma_3 = 0.5
time_step_max = 1000
episodes_1 = sample(MDP, Pi_1, time_step_max, 1000)
episodes_2 = sample(MDP, Pi_2, time_step_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", time_step_max, gamma_3)
rho_2 = occupancy(episodes_2, "s4", "概率前往", time_step_max, gamma_3)
print(rho_1, rho_2)

'''
0.112567796310472 0.23199480615618912
'''