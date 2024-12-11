# -*- coding: utf-8 -*-

import copy
import random
import numpy as np

from rlcards.games.aichess.rlzero.config import CONFIG
from rlcards.games.aichess.rlzero.util.log import error_log


def softmax(x):
	"""
	归一化输出概率
	"""
	prob_s = np.exp(x - np.max(x))
	prob_s /= np.sum(prob_s)
	return prob_s


class TreeNode:
	"""
	todo: 定义叶子节点
	"""
	def __init__(self, parent, prior_p):
		"""
		初始化参数
		"""
		self._parent = parent  # 当前节点父节点
		self._children = {}  # 从动作到TreeNode的映射
		self._n_visits = 0  # 当前当前节点的访问次数
		self._Q = 0  # 当前节点对应动作的平均动作价值
		self._u = 0  # 当前节点的置信上限，P UCT算法
		self._P = prior_p  # 当前节点被选择的先验概率

	def expand(self, action_priors):
		"""
		将不合法的动作概率全部设置为0，通过创建新子节点来展开树
		"""
		for action, prob in action_priors:
			if action not in self._children:
				self._children[action] = TreeNode(self, prob)

	def select(self, c_p_uct):
		"""
		在子节点中选择能够提供最大的Q+U的节点
		返回(action, next_node)的二元组
		"""
		return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_p_uct))

	def get_value(self, c_p_uct):
		"""
		计算并返回此节点的值，它是节点评估Q和此节点的先验的组合
		c_p_uct控制相对影响(0， inf)
		"""
		self._u = (c_p_uct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
		return self._Q + self._u

	def update(self, leaf_value):
		"""
		从叶节点评估中更新节点值
		leaf_value: 这个子节点的评估值来自当前玩家的视角
		"""
		# 统计访问次数
		self._n_visits += 1
		# 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
		self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

	def update_recursive(self, leaf_value):
		"""
		使用递归的方法对所有节点(当前节点对应的支线)进行一次更新
		就像调用update()一样，但是对所有直系节点进行更新
		"""
		# 如果它不是根节点，则应首先更新此节点的父节点
		if self._parent:
			self._parent.update_recursive(-leaf_value)
		self.update(leaf_value)

	def is_leaf(self):
		"""
		检查是否是叶节点，即没有被扩展的节点
		"""
		return self._children == {}

	def is_root(self):
		"""
		检查是否是根节点
		"""
		return self._parent is None


class MCTS:
	"""
	蒙特卡洛搜索
	"""
	def __init__(self, policy_value_fn, c_p_uct=5, n_play_out=2000):
		"""
		初始化属性参数
		policy_value_fn接收board的盘面状态，返回落子概率和盘面评估得分
		"""
		self._root = TreeNode(None, 1.0)
		self._policy = policy_value_fn
		self._c_p_uct = c_p_uct
		self._n_play_out = n_play_out

	def _play_out(self, board):
		"""
		进行一次搜索，根据叶节点的评估值反向更新树节点的参数
		注意，state已经原地进行了修改，因此必须提供副本
		"""
		node = self._root
		while True:
			# 判断是否为叶子节点
			if node.is_leaf():
				# 当搜索结束时，则跳出
				break
			# 贪心搜索下一步动作
			action, node = node.select(self._c_p_uct)
			# 更新棋盘移动动作
			board.do_move(action)
		# 使用网络评估叶子节点，网络输出(动作，概率)元组p的列表以及当前玩家视角的得分[-1, 1]
		action_prob_s, leaf_value = self._policy(board)
		# 查看游戏是否结束
		end, winner = board.game_end()
		if not end:
			# 有些未结束时，则添加动作概率
			node.expand(action_prob_s)
		else:
			# 对于结束状态，将叶子节点的值换成1或-1
			if winner == -1:  # Tie
				leaf_value = 0.0
			else:
				leaf_value = (
					1.0 if winner == board.get_curr_player_id() else -1.0
				)
		# 在本次遍历中更新节点的值和访问次数
		# 必须添加符号，因为两个玩家共用一个搜索树
		node.update_recursive(-leaf_value)

	def get_move_prob_s(self, board, temp=1e-3):
		"""
		按顺序允许所有搜索并返回可用的动作及其对应的概率
		state相当于当前游戏状态，temp介于(0, 1]之间的温度参数
		"""
		for n in range(self._n_play_out):
			board_copy = copy.deepcopy(board)
			self._play_out(board_copy)

		# 根据根节点处的访问计数来计算移动概率
		act_visits = [
			(act, node._n_visits) for act, node in self._root._children.items()
		]
		acts, visits = zip(*act_visits)
		act_prob_s = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
		return acts, act_prob_s

	def update_with_move(self, last_move):
		"""
		在当前的树向前一步，保持已经知道的关于子树的一切
		"""
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0)

	def __str__(self):
		"""
		打印
		"""
		return 'M C T S'