# -*- coding: utf-8 -*-

import time
import copy
import random
import numpy as np

from rlcards.games.aichess.rlzero.env.mcts import MCTS
from rlcards.games.aichess.rlzero.config import CONFIG
from rlcards.games.aichess.rlzero.env.ucci import get_cep_move_func, get_cep_move_func_pika


class MCTSPlayer:
	"""
	基于 M C T S 的AI玩家
	"""
	def __init__(self, policy_value_function, c_p_uct=5, n_play_out=2000, is_self_play=0):
		"""
		初始化蒙特卡洛AI玩家属性参数
		"""
		self.agent = 'AI'
		self.player = None
		self.mct = MCTS(policy_value_function, c_p_uct, n_play_out)
		self._is_self_play = is_self_play
		self.policy_value_function = policy_value_function

	def set_play_ind(self, p):
		"""
		设置玩家
		"""
		self.player = p

	def reset_player(self):
		"""
		重置搜索树
		"""
		self.mct.update_with_move(-1)

	def get_action(self, board, temp=1e-3, return_prob=None, play_with_cep=False):
		"""
		获取移动动作
		如alpha go zero论文一样使用[M C T S]算法返回pi(概率)向量
		"""
		move_prob_s = np.zeros(2086)
		board = copy.deepcopy(board)
		# todo: 象棋引擎选择动作
		if CONFIG['use_cep'] or play_with_cep:
			# print("[CEP-p1]使用象棋引擎来获取合法动作")
			acts = board.legal_moves
			if len(acts) == 1:
				move = acts[0]
				move_prob_s[acts[0]] = 1
				return move, move_prob_s
			start_time = time.time()
			move, lines = get_cep_move_func_pika(board, board.curr_player_color)
			end_time = time.time()
			print("计算动作耗时: ", end_time - start_time)
			if move is None:
				# 将军时，无最优移动策略，则随机选择移动动作
				move = random.choice(board.legal_moves)
				print('Error CEP 象棋引擎[Chess Engine Protocol]')
			move_prob_s[move] = 1
			return move, move_prob_s
		# todo: 深度蒙特卡洛AI玩家选择动作
		else:
			# print("[CEP-p1]使用象棋引擎来获取合法动作")
			acts = board.legal_moves
			if len(acts) == 1:
				move = acts[0]
				move_prob_s[acts[0]] = 1
				return move, move_prob_s
			start_time = time.time()
			move, lines = get_cep_move_func(board, board.curr_player_color)
			end_time = time.time()
			print("计算动作耗时: ", end_time - start_time)
			if move is None:
				# 将军时，无最优移动策略，则随机选择移动动作
				move = random.choice(board.legal_moves)
				print('Error CEP 象棋引擎[Chess Engine Protocol]')
			move_prob_s[move] = 1
			return move, move_prob_s
		# 	# print("[MCT-p2]使用蒙特卡洛树来搜索最佳合法动作")
		# 	acts, prob_s = self.mct.get_move_prob_s(board, temp)
		# 	move_prob_s[list(acts)] = prob_s
		# # 添加dirichlet noise进行搜索(自我对弈需要)
		# if self._is_self_play:
		# 	move = np.random.choice(
		# 		acts,
		# 		p=0.75 * prob_s + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(prob_s)))
		# 	)
		# 	# 更新节点并重用搜索树
		# 	self.mct.update_with_move(move)
		# else:
		# 	move = np.random.choice(
		# 		acts, p=0.95 * prob_s + 0.05 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(prob_s)))
		# 	)
		# 	# 重置根节点
		# 	self.mct.update_with_move(-1)
		# return move, move_prob_s