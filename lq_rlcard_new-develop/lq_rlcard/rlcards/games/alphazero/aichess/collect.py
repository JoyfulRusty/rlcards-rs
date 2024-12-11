# -*- coding: utf-8 -*-

import os
import copy
import time
import pickle

from collections import deque
from rlcards.games.alphazero.aichess import CONFIG
from rlcards.games.alphazero.aichess import MCTSPlayer
from rlcards.games.alphazero.aichess import PolicyValueNet
from rlcards.games.alphazero.aichess.game import Board, Game, move_action2move_id, move_id2move_action, flip_map


class CollectPipeline:
	"""
	对弈收集数据流程
	"""
	def __init__(self):
		"""
		初始化象棋棋牌和游戏逻辑等参数
		"""
		self.iter = 0
		self.temp = 1
		self.board = Board()
		self.game = Game(self.board)
		self.cp_uct = CONFIG['cp_uct']  # uct权重
		self.n_play_out = CONFIG['play_out']  # 每次移动的模拟次数
		self.buffer_size = CONFIG['buffer_size']  # 经验池大小
		self.data_buffer = deque(maxlen=self.buffer_size)

	def load_model(self, model_path=CONFIG['policy_model_path']):
		"""
		加载模型
		"""
		try:
			self.policy_value_net = PolicyValueNet(model_file=model_path)
			print('已加载最新模型')
		except:
			self.policy_value_net = PolicyValueNet()
			print('已加载初始模型')

		# 蒙特卡洛搜索树玩家
		self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn)

	def get_reduce_by_half_data(self, play_data):
		"""
		左右对称变换，扩充数据集一倍，加速一倍训练速度
		"""
		extend_data = []
		# 棋盘状态维度[9, 10, 9]，走子概率 -> 赢家
		for state, mct_prob, winner in play_data:
			# 原始数据
			extend_data.append((state, mct_prob, winner))
			# 水平反转后产生的数据
			state_flip = state.transpose([1, 2, 0])
			state = state.transpose([1, 2, 0])
			for i in range(10):
				for j in range(9):
					state_flip[i][j] = state[i][8 - j]
			state_flip = state_flip.transpose([2, 0, 1])
			mct_prob_flip = copy.deepcopy(mct_prob)
			for i in range(len(mct_prob_flip)):
				mct_prob_flip[i] = mct_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
			extend_data.append((state_flip, mct_prob_flip, winner))
		return extend_data

	def collect_self_play_data(self, n_games=1):
		"""
		收集自我对弈的数据
		"""
		for i in range(n_games):
			self.load_model()  # 加载最新模型
			winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
			play_data = list(play_data)[:]
			self.episode_len = len(play_data)
			# 增加数据
			play_data = self.get_reduce_by_half_data(play_data)
			if os.path.exists(CONFIG['train_data_buffer_path']):
				while True:
					try:
						with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
							data_file = pickle.load(data_dict)
							self.data_buffer = data_file['data_buffer']
							self.iter = data_file['iter']
							del data_file
							self.iter += 1
							self.data_buffer.extend(play_data)
						print('成功载入数据')
						break
					except:
						time.sleep(30)
			else:
				self.data_buffer.extend(play_data)
				self.iter += 1
			data_dict = {'data_buffer': self.data_buffer, 'iter': self.iter}
			with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
				pickle.dump(data_dict, data_file)
		return self.iter

	def run(self):
		"""开始收集数据"""
		try:
			while True:
				iter = self.collect_self_play_data()
				print('batch i: {}, episode_len: {}'.format(iter, self.episode_len))
		except KeyboardInterrupt:
			print('\n\rquit')

collecting_pipeline = CollectPipeline()
collecting_pipeline.run()