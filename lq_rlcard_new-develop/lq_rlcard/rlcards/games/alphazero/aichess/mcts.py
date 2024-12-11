# -*- coding: utf-8 -*-

import copy
import numpy as np

from rlcards.games.alphazero.aichess import CONFIG


def softmax(x):
	"""
	输出概率分布
	"""
	prob_s = np.exp(x - np.max(x))
	prob_s /= np.sum(prob_s)
	return prob_s


class TreeNode:
	"""
	定义叶子节点
	蒙特卡洛搜索树中的节点，树的子节点字典中，键为动作，值为TreeNode
	记录当前节点选择的动作，以及选择该动作后会跳转到的下一个子节点
	每个节点跟踪其自身的Q，先验概率P及其访问次数调整的u
	"""
	def __init__(self, parent, prior_p):
		"""
		初始化参数
		定义节点的父节点和当前节点被选择的先验概率
		"""
		self._Q = 0  # 当前节点对应动作的平均动作价值
		self._u = 0  # 当前节点的置信上限(p)UCT算法
		self._P = prior_p  # 先验概率
		self._parent = parent  # 父节点
		self._children = {}  # 从动作到TreeNode映射
		self._n_visits = 0  # 当前节点的访问次数

	def expand(self, action_priors):
		"""
		通过创建新子节点来展开树
		"""
		for action, prob in action_priors:
			if action not in self._children:
				self._children[action] = TreeNode(self, prob)

	def select(self, cp_uct):
		"""
		在节点中选择能够提供最大的Q+U的节点(action, next_node)的二元组
		"""
		return max(
			self._children.items(),
			key=lambda act_node: act_node[1].get_value(cp_uct)
		)

	def get_value(self, cp_uct):
		"""
		计算并返回此节点的值，它是节点评估Q和此节点的先验的组合，cp_uct控制相对影响(0， inf)
		"""
		self._u = (cp_uct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

		return self._Q + self._u

	def update(self, leaf_value):
		"""
		从节点评估中更新节点值，leaf_value这个子节点的评估值来自当前玩家的视角
		"""
		# 统计访问次数
		self._n_visits += 1
		# 更新Q值，却决于所有访问次数的平均树，使用增量式更新方式
		self._Q = 1.0 * (leaf_value - self._Q) / self._n_visits

	def update_recursive(self, leaf_value):
		"""
		使用递归的方法对所有节点(当前节点对应的支线)进行一次更新
		如调用update()一样，但它是对所有直系节点进行更新
		"""
		# 当不是根节点，则应首先更新此节点的父节点
		if self._parent:
			self._parent.update_recursive(-leaf_value)
		self.update(leaf_value)

	def is_leaf(self):
		"""
		检查是否为叶节点，即没有被扩展的节点
		"""
		return self._children == {}

	def is_root(self):
		"""
		判断是否为根
		"""
		return self._parent is None

class MCTS:
	"""
	蒙特卡洛搜索树
	"""
	def __init__(self, policy_value_fn, cp_uct=5, n_play_out=2000):
		"""
		接收board的盘面状态，返回落子概率和盘面评估得分
		"""
		self._root = TreeNode(None, 1.0)
		self._policy = policy_value_fn
		self._cp_out = cp_uct
		self._n_play_out = n_play_out

	def _play_out(self, state):
		"""
		进行一次搜索，根据叶节点的评估值进行反向传递更新树节点的参数
		注意: state已就地修改，因此必须提供副本
		"""
		node = self._root
		while True:
			if node.is_leaf():
				break
			# 贪心算法策略，计算下一步动作
			action, node = node.select(self._cp_out)
			state.do_move(action)

		# 使用网络评估叶子节点，网络输出[动作，概率]元组的p列表
		# 以及当前玩家视角的得分[-1, 1]
		action_prob_s, leaf_value = self._policy(state)
		# 判断游戏是否结束
		end, winner = state.game_end()
		if not end:
			node.expand(action_prob_s)
		else:
			# 结束状态，则将叶子节点的值换成1或-1
			if winner == -1:  # Tie
				leaf_value = 0.0
			else:
				leaf_value = 1.0 if winner == state.get_current_player_id() else -1.0

		# 在本次变量中更新节点的值和访问次数
		# 必须添加符号，因为两个玩家公用一个搜索树
		node.update_recursive(-leaf_value)

	def get_move_prob(self, state, temp=1e-3):
		"""
		按顺序运行所有搜索并返回可用的动作及其相应的概率
		state表示当前游戏的状态，temp表示介于(0, 1]之间的温度参数
		"""
		for n in range(self._n_play_out):
			state_copy = copy.deepcopy(state)
			self._play_out(state_copy)

		# 根据根系欸但的访问计数来计算移动概率
		act_visits = [
			(act, node._n_visits) for act, node in self._root._children.items()
		]

		acts, visits = zip(*act_visits)
		act_prob_s = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

		return acts, act_prob_s

	def update_with_move(self, last_move):
		"""
		在当前的树向前一步，保持已经直到的关于子树的一切
		"""
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0)


class MCTSPlayer:
	"""
	基于M C T S的AI玩家
	"""
	def __init__(self, policy_value_function, cp_uct=5, n_play_out=2000, is_self_play=0):
		"""
		初始化蒙特卡洛搜索树参数
		"""
		self.agent = 'AI'
		self._is_self_play = is_self_play
		self.mct_s = MCTS(policy_value_function, cp_uct, n_play_out)

	def reset_player(self):
		"""
		重置搜索树
		"""
		self.mct_s.update_with_move(-1)

	def get_action(self, board, temp=1e-3, return_prob=0):
		"""
		与alphaGo_Zero论文一样使用MCTS算法返回的pi向量
		"""
		move_prob_s = np.zeros(2086)
		acts, prob_s = self.mct_s.get_move_prob(board, temp)
		move_prob_s[list(acts)] = prob_s
		if self._is_self_play:
			# 添加Dirichlet Noise进行探索(自我对弈需要)
			move = np.random.choice(
				acts,
				p=0.75 * prob_s + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(prob_s)))
			)
			# 更新根节点并重用搜索树
			self.mct_s.update_with_move(move)
		else:
			# 使用默认的temp=1e-3，它几乎相当于选择具有最高概率的移动
			move = np.random.choice(acts, p=prob_s)
			# 重置根节点
			self.mct_s.update_with_move(-1)
		if return_prob:
			return move, move_prob_s
		else:
			return move