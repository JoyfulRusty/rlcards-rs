# -*- coding: utf-8 -*-

import math
import numpy as np

from copy import deepcopy
from rlcards.games.alphazero.config import CFG


class TreeNode(object):
	"""
   todo: 表示板状态并存储该状态下操作的统计信息

	Nsa: 访问计数的整数
	Wsa: 总操作值的浮点数
	Qsa: 平均操作值的浮点数
	Psa: 到达此节点的先验概率的浮点数
	action: 到达此节点的先前移动的元组（行，列）
	children: 存储子节点的列表
	child_pro: 包含子概率的向量
	parent: 表示父节点的树节点
    """

	def __init__(self, parent=None, action=None, psa=0.0, child_pro=None):
		"""
		使用初始统计信息和数据对TreeNode进行初始化
		"""
		self.Nsa = 0
		self.Wsa = 0.0
		self.Qsa = 0.0
		self.Psa = psa
		self.action = action
		self.children = []
		self.child_pro = child_pro or []
		self.parent = parent

	def is_not_leaf(self):
		"""
		todo: 检查树节点是否是叶子
		返回一个布尔值，指示树节点是否为叶子
		"""
		if len(self.children) > 0:
			return True
		return False

	def select_child(self):
		"""
		todo: 根据AlphaZero P U C T公式选择一个子节点
		返回根据PUCT最有前途的子TreeNode
		"""
		c_puct = CFG.c_puct

		highest_uct = 0
		highest_index = 0

		# 选择具有最高 Q + U 值的子项
		for idx, child in enumerate(self.children):
			uct = child.Qsa + child.Psa * c_puct * (
					math.sqrt(self.Nsa) / (1 + child.Nsa))
			if uct > highest_uct:
				highest_uct = uct
				highest_index = idx

		return self.children[highest_index]

	def expand_node(self, game, psa_vector):
		"""
		通过将有效移动添加为子节点来扩展当前节点
		psa_vector,包含每次移动的移动概率的列表
		"""
		self.child_psas = deepcopy(psa_vector)
		valid_moves = game.get_valid_moves(game.current_player)
		for idx, move in enumerate(valid_moves):
			if move[0] is not 0:
				action = deepcopy(move)
				self.add_child_node(
					parent=self,
					action=action,
					psa=psa_vector[idx]
				)

	def add_child_node(self, parent, action, psa=0.0):
		"""
		创建子树节点并将其添加到当前节点
		"""

		child_node = TreeNode(parent=parent, action=action, psa=psa)
		self.children.append(child_node)
		return child_node

	def back_prop(self, wsa, v):
		"""
		根据游戏结果更新当前节点的统计信息
		wsa: 表示此状态的操作值的浮点数
		v: 表示此状态的网络值的浮点数
		"""
		self.Nsa += 1
		self.Wsa = wsa + v
		self.Qsa = self.Wsa / self.Nsa


class MonteCarloTreeSearch(object):
	"""
	todo: 表示蒙特卡罗树搜索算法

	root: 表示板状态及其统计信息的树节点
	game: 包含游戏状态的对象
	net: 包含神经网络的对象
	"""

	def __init__(self, net):
		"""
		todo: 使用 TreeNode、Board 和神经网络初始化 TreeNode
		"""
		self.root = None
		self.game = None
		self.net = net

	def search(self, game, node, temperature):
		"""
		todo: M C T S 循环以获得可以在给定状态下播放的最佳动作

		game: 包含游戏状态的对象
		node: 表示板状态及其统计信息的树节点
		temperature: 用于控制探索水平的浮子

		一个子节点, 表示在此状态下播放的最佳移动
		"""
		self.root = node
		self.game = game

		for i in range(CFG.num_mcts_sims):
			node = self.root
			game = self.game.clone()  # 为每个循环创建一个新克隆

			# 当节点不是叶子时循环
			while node.is_not_leaf():
				node = node.select_child()
				game.play_action(node.action)

			# 从网络中获取此状态的移动概率和值
			psa_vector, v = self.net.predict(game.state)

			# 将狄利克雷噪声添加到根节点的psa_vector
			if node.parent is None:
				psa_vector = self.add_dirichlet_noise(game, psa_vector)

			valid_moves = game.get_valid_moves(game.current_player)
			for idx, move in enumerate(valid_moves):
				if move[0] is 0:
					psa_vector[idx] = 0

			psa_vector_sum = sum(psa_vector)

			# 重整化PSA载体
			if psa_vector_sum > 0:
				psa_vector /= psa_vector_sum

			# 尝试扩展当前节点
			node.expand_node(game=game, psa_vector=psa_vector)

			game_over, wsa = game.check_game_over(game.current_player)

			# 将节点统计信息反向传播到根节点
			while node is not None:
				wsa = -wsa
				v = -v
				node.back_prop(wsa, v)
				node = node.parent

		highest_nsa = 0
		highest_index = 0

		# 使用温度参数选择孩子的动作
		for idx, child in enumerate(self.root.children):
			temperature_exponent = int(1 / temperature)

			if child.Nsa ** temperature_exponent > highest_nsa:
				highest_nsa = child.Nsa ** temperature_exponent
				highest_index = idx

		return self.root.children[highest_index]

	def add_dirichlet_noise(self, game, psa_vector):
		"""
		todo: 将狄利克雷噪声添加到根节点的psa_vector

		这是为了进一步探索

		game: 包含游戏状态的对象
		psa_vector: 概率向量

		添加了狄利克雷噪声的概率向量
		"""
		dirichlet_input = [CFG.dirichlet_alpha for x in range(game.action_size)]

		dirichlet_list = np.random.dirichlet(dirichlet_input)
		noisy_psa_vector = []

		for idx, psa in enumerate(psa_vector):
			noisy_psa_vector.append(
				(1 - CFG.epsilon) * psa + CFG.epsilon * dirichlet_list[idx])

		return noisy_psa_vector