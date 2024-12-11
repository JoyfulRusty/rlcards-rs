# -*- coding: utf-8 -*-

import time
import asyncio
import numpy as np

from env.game_state import GameState

def softmax(x):
	"""
	输出分布概率
	"""
	prob = np.exp(x - np.max(x))
	prob /= np.sum(prob)
	return prob

class TreeNode:
	"""
	M C T S树中的一个节点
	每个节点跟踪自己的值Q、先验概率P及其访问计数调整的先验得分u
	"""
	def __init__(self, parent, prior_p, noise=False):
		"""
		初始化节点属性参数
		"""
		self._parent = parent
		self._children = {}  # 从操作到树节点的映射
		self._n_visits = 0  # 访问次数
		self._Q = 0
		self._u = 0
		self._P = prior_p
		self.virtual_loss = 0
		self.noise = noise

	def expand(self, action_priors):
		"""
		通过创建新的子项来扩展树
		action_priors为根据策略函数的操作元组及其先验概率的列表
		"""
		# 每个选择操作时都应应用狄利克雷噪声
		if False and self.noise is True and self._parent is None:
			noised_d = np.random.dirichlet([0.3] * len(action_priors))
			for (action, prob), one_noise in zip(action_priors, noised_d):
				if action not in self._children:
					# 添加噪音来控制动作概率分布
					prob = (1 - 0.25) * prob + 0.25 * one_noise
					self._children[action] = TreeNode(self, prob, noise=self.noise)
		else:
			for action, prob in action_priors:
				if action not in self._children:
					self._children[action] = TreeNode(self, prob)

	def select(self, c_p_uct):
		"""
		在子项中选择操作，该操作提供最大操作值Q加上奖励u(P)
		tuple -> (action, next_node)
		"""
		if self.noise is False:
			return max(
				self._children.items(), key=lambda act_node: act_node[1].get_value(c_p_uct)
			)
		elif self.noise is True and self._parent is not None:
			return max(
				self._children.items(),
				key=lambda act_node: act_node[1].get_value(c_p_uct)
			)
		else:
			noise_d = np.random.dirichlet([0.3] * len(self._children))
			return max(
				list(zip(noise_d, self._children.items())),
				key=lambda act_node: act_node[1][1].get_value(c_p_uct, noise_p=act_node[0])
			)[1]

	def update(self, leaf_value):
		"""
		从叶评估更新节点值
		leaf_value为从当前玩家的角度看子树评估的价值
		"""
		# 统计访问次数
		self._n_visits += 1
		# 更新Q，所有访问值的运行平均值
		self._Q = 1.0 * (leaf_value - self._Q) / self._n_visits

	def update_recursive(self, leaf_value):
		"""
		就像对update()的调用一样，但对所有祖先递归应用
		"""
		# 如果它不是根节点，则应首先更新此节点的父节点
		if self._parent:
			self._parent.update_recursive(-leaf_value)
		self.update(leaf_value)

	def get_value(self, c_p_uct, noise_p=None):
		"""
		计算并返回此节点的值
		它是叶评估Q的组合，并且此节点的先前已针对其访问计数u进行了调整
		c_p_uct为(0, inf)中的一个数字，控制值Q和先验概率P对此节点分数的相对影响
		"""
		if noise_p is None:
			self._u = (
					c_p_uct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
			)
			return self._Q + self._u + self.virtual_loss
		else:
			self._u = (
					c_puct * (self._P * 0.75 + noise_p * 0.25) * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
			)
			return self._Q + self._u + self.virtual_loss

	def is_leaf(self):
		"""
		检查叶节点(即此节点下没有节点被扩展)
		"""
		return self._children == {}

	def is_root(self):
		"""
		判断是否为根节点
		"""
		return self._parent is None

class MCTS:
	"""
	蒙特卡洛树搜索
	"""
	def __init__(
			self,
			policy_value_fn,
			c_p_uct=5,
			n_play_out=10000,
			search_threads=2,
			virtual_loss=3,
			policy_loop_arg=False,
			d_noise=False,
			play=False):
		"""
		policy_value_fn为一个函数，它接受棋盘状态并输出当前玩家的(action: 动作, probability: 概率)元组列表以及[-1， 1]中的分数
		即从当前玩家的角度来看，最终游戏分数的期望值
		c_p_uct(0, INF)中的一个数字，用于控制探索收敛到最大值策略的速度。更高的值意味着对先验的依赖更多
		"""
		self._root = TreeNode(None, 1.0, noise=d_noise)
		self._policy = policy_value_fn
		self._c_p_uct = c_p_uct
		self._n_play_out = n_play_out
		self.virtual_loss = virtual_loss
		self.loop = asyncio.get_event_loop()
		self.policy_loop_arg = policy_loop_arg
		self.sem = asyncio.Semaphore(search_threads)
		self.new_expanding = set()
		self.select_time = 0
		self.policy_time = 0
		self.update_time = 0
		self.num_proceed = 0
		self.d_noise = d_noise
		self.play = play

	async def _play_out(self, state_input):
		"""
		运行从根到叶的单个播放，在叶上获取值并将其传播回其父级。状态是就地修改的，因此必须提供副本
		"""
		state = GameState()
		state.copy_custom(state_input)
		with await self.sem:
			node = self._root
			road = []
			while 1:
				while node in self.now_expanding:
					await asyncio.sleep(1e-4)
				start = time.time()
				if node.is_leaf():
					break
				# 贪婪策略选择下一步动作
				action, node = node.select(self._c_p_uct)
				road.append(node)
				node.virtual_loss -= self.virtual_loss
				state.do_move(action)
				self.select_time += (time.time() - start)
			# 在离开节点时，如果长检查或长捕获则切断节点
			if state.should_cutoff() and not self.play:
				# 切断节点
				for one_node in road:
					one_node.virtual_loss += self.virtual_loss
				# 现在这个时候，不更新整个树分支，精度损失应该很小node.update_recursive(-leaf_value)将虚拟损失设置为-INF，
				# 这样其他线程就不会再访问同一个节点(所以节点被切断了)
				node.virtual_loss = -np.inf
				# node.update_recursive(leaf_value)
				self.update_time += (time.time() - start)
				# 进程数量仍然+1
				self.num_proceed += 1
				return
			start = time.time()
			self.now_expanding.add(node)
			# 使用网络评估叶子，该网络输出当前玩家的(动作，概率)元组p列表以及[-1,1]中的分数v
			if self.policy_loop_arg is False:
				action_prob, leaf_value = await self._policy(state)
			else:
				action_prob, leaf_value = await self._policy(state, self.loop)
			state = time.time()
			# 检查游戏是否结束
			end, winner = state.game_end()
			if not end:
				node.expand(action_prob)
			else:
				# 对于结束状态，返回“true”leaf_value
				if winner == -1:  # tie
					leaf_value = 0.0
				else:
					leaf_value = (1.0 if winner == state.get_current_player() else -1.0)
			# 更新此遍历中节点的值和访问计数
			for one_node in road:
				one_node.virtual_loss += self.virtual_loss
			node.update_recursive(-leaf_value)
			self.now_expanding.remove(node)
			# node.update_recursive(leaf_value)
			self.update_time += (time.time() - start)
			self.num_proceed += 1

	def get_move_prob(
			self,
			state,
			temp=1e-3,
			predict_workers=None,
			can_apply_d_noise=False,
			verbose=False,
			infer_mode=False,
			no_act=None):
		"""
		按顺序运行所有播出，并返回可用操作及其相应的概率
		state为当前游戏状态
		temp为温度参数(0,1)控制探索级别
		"""
		# 判断预测workers是否为None
		predict_workers = predict_workers or []
		if can_apply_d_noise is False:
			self._root.noise = False
		coroutine_list = []
		for n in range(self._n_play_out):
			# state_copy = copy.deepcopy(state)
			coroutine_list.append(self._playout(state))
		coroutine_list += predict_workers
		self.loop.run_until_complete(asyncio.gather(*coroutine_list))
		# 根据根节点上的访问计数计算移动概率
		act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
		# todo: 冻掉次数?
		if not act_visits:
			if infer_mode:
				return None, None, None
			return None, None
		acts, visits = zip(*act_visits)
		visits = np.array(visits)
		if no_act:
			for act_index in range(len(acts)):
				if acts[act_index] in no_act:
					visits[act_index] = 0
		act_prob = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
		info = []
		if infer_mode:
			info = [(act, node._n_visits, node._Q, node._P) for act, node in self._root._children.items()]
		if infer_mode:
			return acts, act_prob, info
		else:
			return acts, act_prob

	def update_with_move(self, last_move, allow_legacy=True):
		"""
		在树中向前迈进，保留已经知道的关于子树的一切
		"""
		self.num_proceed = 0
		if last_move in self._root._children and allow_legacy:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0, noise=self.d_noise)

	def __str__(self):
		"""
		打印
		"""
		return "M C T S"