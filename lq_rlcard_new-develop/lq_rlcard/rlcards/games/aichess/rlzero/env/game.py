# -*- coding: utf-8 -*-

import random
import numpy as np

from rlcards.games.aichess.rlzero.const import const
from rlcards.games.aichess.rlzero.config import CONFIG
from rlcards.games.aichess.rlzero.util.util import move_id2move_action, move_action2move_id


class Game:
	"""
	todo: 游戏流程类
	"""
	def __init__(self, board):
		"""
		初始化游戏属性
		"""
		self.board = board
		self.play_with_cep = CONFIG['play_with_cep']  # 象棋引擎协议[u c c i]

	def start_play(self, player1, player2, process_id=None, winner_dict=None):
		"""
		todo: 用于人机对战、双人对战
		"""
		# 初始化棋盘
		self.board.init_board()
		states, mct_prob_s, curr_players, moves = [], [], [], []
		p1, p2 = 1, 2
		player1.set_player_ind(1)
		player2.set_player_ind(2)
		players = {p1: player1, p2: player2}
		while True:
			curr_player = self.board.get_curr_player_id()  # 红子对应的玩家id
			player_policy_func = players[curr_player]  # 决定当前玩家的代理
			move, move_prob_s = player_policy_func.get_action(self.board)  # 当前玩家代理拿到动作
			states.append(self.board.get_curr_state())
			mct_prob_s.append(move_prob_s)
			curr_players.append(self.board.curr_player_id)
			moves.append(move_id2move_action[move])
			self.board.do_move(move)
			end, winner = self.board.game_end()
			if end:
				winner_z = np.zeros(len(curr_players))
				if winner != -1:
					winner_z[np.array(curr_players) == winner] = 1.0
					winner_z[np.array(curr_players) != winner] = -1.0
					if winner_dict:
						final_winner = winner_dict[winner]
					else:
						final_winner = players[winner]
					print(f"Game end. Winner is {final_winner}, Process_id: {process_id}")
				else:
					print(f"Game end. Tie, Process_id: {process_id}")
				log_info = f'ai vs ai: evaluate'
				return winner, zip(states, mct_prob_s, winner_z), moves, log_info

	def start_self_play(self, player, is_shown=False, temp=1e-3, cep_pos=1):
		"""
		todo: 自我对弈
		使用蒙特卡洛树搜索开始自我对弈，存储游戏状态(状态，蒙特卡洛落子概率，胜负手)三元组用于神经网络训练
		初始化棋盘, start_player=1
		"""
		self.board.init_board()
		states, mct_prob_s, curr_players, moves = [], [], [], []
		# 开始自我对弈
		_count = 0
		# todo: 一个mct卡洛玩家模型，但红黑下棋方调用方法不同
		while True:
			if self.play_with_cep and _count % 2 == cep_pos:
				play_with_cep = True
			else:
				play_with_cep = False
			move, move_prob_s = player.get_action(
				self.board,
				temp=temp,
				return_prob=1,
				play_with_cep=play_with_cep
			)
			_count += 1
			# 保存自我对弈的数据
			states.append(self.board.get_curr_state())
			mct_prob_s.append(move_prob_s)
			curr_players.append(self.board.curr_player_id)
			moves.append(move_id2move_action[move])

			# 打印观测
			board_map, color_cur = self.board.curr_map()
			print(f"玩家: {self.board.curr_player_id}, 棋色: {self.board.curr_player_color}, 是否使用cep: {play_with_cep}")
			print("输出移动动作: ", move_id2move_action[move])
			for i in board_map:
				print(i)
			print('\n')

			# 执行一步落子
			self.board.do_move(move)
			end, winner = self.board.game_end()

			if end:
				# 从每一个状态state对应的玩家的视角保存胜负信息
				winner_z = np.zeros(len(curr_players))
				if winner != -1:
					winner_z[np.array(curr_players) == winner] = 1.0
					winner_z[np.array(curr_players) != winner] = -1.0
				# 重置蒙特卡洛根节点
				player.reset_player()
				if is_shown:
					if winner != -1:
						print("Game end. Winner is:", winner)
					else:
						print('Game end. Tie')
				log_info = f'ai vs ai: self'
				if self.play_with_cep:
					if winner != cep_pos + 1:
						log_info = 'ai vs cep: ai 赢了'
					else:
						log_info = 'ai vs cep: ai 输了'
				return winner, zip(states, mct_prob_s, winner_z), moves, log_info