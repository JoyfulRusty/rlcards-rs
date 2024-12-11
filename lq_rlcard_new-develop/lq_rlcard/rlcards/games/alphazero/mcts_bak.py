# -*- coding: utf-8 -*-

import sys
import math
import random

AVAILABLE_CHOICES = [1, -1, 2, -2]  # 可选项
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)  # 可用选择编号
MAX_ROUND_NUMBER = 20000  # 最大轮数

class State:
	"""
	todo: 蒙特卡罗树搜索的游戏状态
	记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round树、从开始到当前的执行记录
	需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作
	"""

	def __init__(self):
		"""
		todo: 初始化参数
		对于第一个根节点，索引为0，游戏应从1开始
		"""
		self.current_value = 0.0
		self.current_round_index = 0
		self.cumulative_choices = []  # 累积选择

	def get_current_value(self):
		"""
		todo: 当前价值
		"""
		return self.current_value

	def set_current_value(self, value):
		"""
		todo: 设置当前价值
		"""
		self.current_value = value

	def get_current_round_index(self):
		"""
		todo: 获取当前轮此
		"""
		return self.current_round_index

	def set_current_round_index(self, turn):
		"""
		todo: 设置当前轮次
		"""
		self.current_round_index = turn

	def get_cumulative_choices(self):
		"""
		todo: 获取当前累积奖励
		"""
		return self.cumulative_choices

	def set_cumulative_choices(self, choices):
		"""
		todo: 设置当前累积奖励
		"""
		self.cumulative_choices = choices

	def is_terminal(self):
		"""
		todo: 轮数指数从1到最大轮数
		"""
		if self.current_round_index == MAX_ROUND_NUMBER:
			return True
		return False

	def compute_reward(self):
		"""
		todo: 计算奖励
		"""
		return -abs(1 - self.current_value)

	def get_next_state_with_random_choice(self):
		"""
		todo: 通过随机选择获取下一个状态
		"""
		random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])
		next_state = State()
		next_state.set_current_value(self.current_value + random_choice)
		next_state.set_current_round_index(self.current_round_index + 1)
		next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])

		return next_state

	def __repr__(self):
		"""
		todo: 打印
		"""
		return "State: {}, value: {}, round: {}, choices: {}".format(
			hash(self),
			self.current_value,
			self.current_round_index,
			self.cumulative_choices
		)

class Node:
	"""
	todo: 蒙特卡罗树的树结构Node
	包含了父节点和直接点等信息，还用于计算UCB的遍历次数和QUALITY值
	还有游戏选择这个Node的State
	"""

	def __init__(self):
		"""
		todo: 初始化参数
		"""
		self.state = None
		self.parent = None
		self.children = []
		self.visit_times = 0
		self.quality_value = 0.0

	def set_state(self, state):
		"""
		todo: 设置状态
		"""
		self.state = state

	def get_state(self):
		"""
		todo: 获取状态
		"""
		return self.state

	def get_parent(self):
		"""
		todo: 获取父亲节点
		"""
		return self.parent

	def set_parent(self, parent):
		"""
		todo: 设置父亲节点
		"""
		self.parent = parent

	def get_children(self):
		"""
		todo: 获取孩子节点
		"""
		return self.children

	def get_visit_times(self):
		"""
		todo: 获取查看次数
		"""
		return self.visit_times

	def set_visit_times(self, times):
		"""
		todo: 设置查看次数
		"""
		self.visit_times = times

	def visit_time_add_one(self):
		"""
		todo: 添加一次查看次数
		"""
		self.visit_times += 1

	def get_quality_value(self):
		"""
		todo: 获取数量价值
		"""
		return self.quality_value

	def set_quality_value(self, value):
		"""
		todo: 设置数量价值
		"""
		self.quality_value = value

	def quality_value_add_n(self, n):
		"""
		todo: 添加一次价值
		"""
		self.quality_value += n

	def is_all_expand(self):
		"""
		todo: 是否全部展开
		"""
		return len(self.children) == AVAILABLE_CHOICE_NUMBER

	def add_child(self, sub_node):
		"""
		todo: 添加子节点
		"""
		sub_node.set_parent(self)
		self.children.append(sub_node)

	def __repr__(self):
		"""
		打印
		"""
		return "Node: {}, Q/N: {}/{}, state: {}".format(
			hash(self),
			self.quality_value,
			self.visit_times,
			self.state
		)

def tree_policy(node):
	"""
	todo: 蒙特卡罗树的Selection和Expansion阶段
	传入当前需要开始搜索的节点，例如，根节点，根据exploration/exploitation算法返回最好的需要expend的节点[当节点为叶子节点则直接返回]
	基本策略是先找到为选择过的子节点，如果有多个则随机选择，如果都选择过就找exploration/exploitation的UCB值最大的，如果UCB值相等则随机选
	"""

	# 检查当前节点是否为叶节点
	while not node.get_state().is_terminal():
		if node.is_all_expand():
			node = best_child(node, True)
		else:
			# 返回一个新的子节点
			sub_node = expand(node)
			return sub_node

	# 返回叶节点
	return node

def default_policy(node):
	"""
	todo: 默认策略，蒙特卡洛树搜索的Simulation阶段
	输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward
	注意输入的节点应该不是子节点，而是有未执行的Action可以expend的

	基本策略是随机选择Action
	"""
	if not node:
		return -1
	# 获取游戏状态
	current_state = node.get_state()

	# 直到游戏结束返回
	while not current_state.is_terminal():
		# 选择一个随机动作进行播放并获得下一个状态
		current_state = current_state.get_next_state_with_random_choice()
	final_state_reward = current_state.compute_reward()

	return final_state_reward

def expand(node):
	"""
	todo: 扩展新节点
	输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点
	注意，需要保证新增的节点与其他节点Action不同
	"""
	tried_sub_node_states = [sub_node.get_state() for sub_node in node.get_children()]
	new_state = node.get_state().get_next_state_with_random_choice()
	# 检查直到获得具有与其他状态不同的操作的新状态
	while new_state in tried_sub_node_states:
		new_state = node.get_state().get_next_state_with_random_choice()

	# 添加一个子节点
	sub_node = Node()
	sub_node.set_state(new_state)
	node.add_child(sub_node)

	return sub_node

def best_child(node, is_exploration):
	"""
	todo: 使用UCB算法，选出当前Q值得分最高的子节点
	"""

	# 使用最小浮点数
	best_score = -sys.maxsize
	best_sub_node = None

	# 遍历所有子节点以找到最佳节点
	for sub_node in node.get_children():
		# 忽略探索以进行推理
		if is_exploration:
			c = 1 / math.sqrt(2.0)
		else:
			c = 0.0

		# UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
		left = sub_node.get_quality_value() / sub_node.get_visit_times()
		right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
		score = left + c * math.sqrt(right)

		# 找出Q值最高的子节点
		if score > best_score:
			best_sub_node = sub_node
			best_score = score

	return best_sub_node

def backup(node, reward):
	"""
	todo: 反向传播
	蒙特卡罗搜索的Backpropagation阶段
	输入前面获取需要expend的节点和新执行Action和reward，反馈给expend节点和上游所有节点并更新对应数据
	"""

	# 更新使用根节点
	while node is not None:
		# 更新查看次数
		node.visit_time_add_one()
		# 更新数量价值
		node.quality_value_add_n(reward)
		# 将节点更改为父节点
		node = node.parent

def monte_carlo_tree_search(node):
	"""
	todo: 蒙特卡洛搜索算法步骤
	蒙特卡洛素算法树搜索算法，传入一个根节点，在有限的时间内根据已经搜索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点
	蒙特卡洛算法树搜索包含四个步骤: Selection、Expansion、Simulation、Backpropagation
	1.Selection和Expansion步骤使用tree policy直到值的探索的节点
	2.Simulation步骤使用default policy也就是选中的节点上随机算法选一个子节点并计算reward
	3.Backpropagation，也就是把reward更新到所有经过的选中节点的节点上

	进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点
	"""

	# 计算预算
	computation_budget = 20

	# 在计算预算下尽可能多地运行
	for i in range(computation_budget):
		# 找到要扩展的最佳节点
		expand_node = tree_policy(node)
		# 随机运行添加节点并获得奖励
		reward = default_policy(expand_node)
		# 用奖励更新所有通过的节点
		backup(expand_node, reward)

	# 获得最佳的下一个节点
	best_next_node = best_child(node, False)

	return best_next_node

def main():
	"""
	todo: 运行主函数
	创建初始化状态和初始化节点
	"""
	init_state = State()
	init_node = Node()
	init_node.set_state(init_state)
	current_node = init_node

	# 设置回合
	for i in range(20000):
		print("Play round: {}".format(i + 1))
		current_node = monte_carlo_tree_search(current_node)
		print("Choose node: {}".format(current_node))

if __name__ == "__main__":
	main()