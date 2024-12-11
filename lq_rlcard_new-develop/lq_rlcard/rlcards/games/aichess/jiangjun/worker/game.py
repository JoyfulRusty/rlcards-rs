# -*- coding: utf-8 -*-

import os
import time
import random
import logging
import numpy as np
import tensorflow as tf

from lib import cbf
from config import conf
from agent import resnet, players
from env.game_state import GameState
from env.cchess_env import create_uci_labels
from lib.utils import get_latest_weight_path

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s][%(message)s]",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )

running_time = 0
running_step = 0


def count_piece(state_str):
	"""
	统计棋子
	"""
	piece_set = {
		'A',
		'B',
		'C',
		'K',
		'N',
		'P',
		'R',
		'a',
		'b',
		'c',
		'k',
		'n',
		'p',
		'r'
	}

	return sum([1 for single_chessman in state_str if single_chessman in piece_set])


class Game:
	"""
	游戏
	"""
	def __init__(self, white, black, verbose=True):
		"""
		初始化游戏参数
		"""
		self.white = white
		self.black = black
		self.verbose = verbose
		self.game_state = GameState()
		self.total_time = 0
		self.steps = 0

	def play_till_end(self):
		"""
		玩到最后
		"""
		global running_step
		global running_time
		moves = []
		peace_round = 0
		remain_piece = count_piece(self.game_state.state_str)
		while True:
			start_time = time.time()
			if self.game_state.move_number % 2 == 0:
				player_name = 'w'
				player = self.white
				opponent_player = self.black
			else:
				player_name = 'b'
				player = self.black
				opponent_player = self.white

			move, score = player.make_move(self.game_state, allow_legacy=True)
			opponent_player.oppoent_make_move(move, allow_legacy=True)

			if move is None:
				winner = 'b' if player_name == 'w' else 'w'
				break
			moves.append(move)
			# if self.verbose:
			total_time = time.time() - start_time
			self.total_time += total_time
			self.steps += 1
			running_time += total_time
			running_step += 1
			logging.info('time average {}'.format(round(running_time / running_step, 2)))
			logging.info('move {} {} play {} score {} use {:.2f}s pr {} pid {}'.format(
				self.game_state.move_number,
				player_name,
				move,
				score if player_name == 'w' else -score,
				total_time,
				peace_round,
				os.getpid())
			)
			game_end, winner_p = self.gamestate.game_end()
			if game_end:
				winner = winner_p
				break
			remain_piece_round = count_piece(self.game_state.statestr)
			if remain_piece_round < remain_piece:
				remain_piece = remain_piece_round
				peace_round = 0
			else:
				peace_round += 1
			if peace_round > conf.SelfPlayConfig.non_cap_draw_round:
				winner = 'peace'
				break
		return winner, moves


class NetworkPlayGame(Game):
	"""
	网络连接游戏
	"""
	def __init__(self, network_w, network_b, **xargs):
		"""
		初始化参数
		"""
		white_player = players.NetworkPlayer('w', network_w, **xargs)
		black_player = players.NetworkPlayer('b', network_b, **xargs)
		super(NetworkPlayGame, self).__init__(white_player, black_player)


class ContinousNetworkPlayGames:
	"""
	连续网络游戏
	"""
	def __init__(
			self,
			network_w=None,
			network_b=None,
			white_name='net',
			black_name='net',
			random_switch=True,
			record_game=True,
			record_dir='data/distributed/',
			play_times=np.inf,
			distributed_dir='data/prepare_weight',
			**xargs):
		"""
		初始化参数
		"""
		self.network_w = network_w
		self.network_b = network_b
		self.white_name = white_name
		self.black_name = black_name
		self.random_switch = random_switch
		self.play_times = play_times
		self.record_game = record_game
		self.record_dir = record_dir
		self.xargs = xargs
		self.distributed_dir = distributed_dir

	def begin_of_game(self):
		"""
		游戏开始
		"""
		pass

	def end_of_game(self, cbf_name, moves, cb_file, training_dt, epoch):
		"""
		游戏结束
		"""
		pass

	def play(self, data_url=None, epoch=0, yun_dao_new_data_dir=None):
		"""
		开始玩游戏
		"""
		num = 0
		while num < self.play_times:
			time_one_game_start = time.time()
			num += 1
			self.begin_of_game()
			if self.random_switch and random.random() < 0.5:
				self.network_w, self.network_b = self.network_b, self.network_w
				self.white_name, self.black_name = self.black_name, self.white_name
			network_play_game = NetworkPlayGame(self.network_w, self.network_b, **self.xargs)
			winner, moves = network_play_game.play_till_end()
			stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
			date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
			cb_file = cbf.CBF(
				black=self.black_name,
				red=self.white_name,
				date=date,
				site='北京',
				name='noname',
				datemodify=date,
				redteam=self.white_name,
				blackteam=self.black_name,
				round='第一轮'
			)
			cb_file.receive_moves(moves)
			rand_stamp = random.randint(0, 1000)
			cb_file_name = '{}_{}_mcts-mcts_{}-{}_{}.cbf'.format(
				stamp, rand_stamp, self.white_name, self.black_name, winner)
			cbf_name = os.path.join(self.record_dir, cb_file_name)
			cbfile.dump(cbf_name)
			if data_url:
				output_game_file_path = os.path.join(data_url, cb_file_name)
				cbfile.dump(output_game_file_path)
			if yundao_new_data_dir:
				import moxing as mox
				mox.file.copy(cbf_name, os.path.join(yun_dao_new_data_dir, cb_file_name))
			training_dt = time.time() - time_one_game_start
			self.end_of_game(cb_file_name, moves, cb_file, training_dt, epoch)


class DistributedSelfPlayGames(ContinousNetworkPlayGames):
	"""
	分布式自玩游戏
	"""
	def __init__(self, gpu_num=0, auto_update=True, **kwargs):
		"""
		初始化参数
		"""
		self.gpu_num = gpu_num
		self.auto_update = auto_update
		self.model_name_in_use = None  # 用于追踪最新权重
		super(DistributedSelfPlayGames, self).__init__(**kwargs)

	def begin_of_game(self):
		"""
		开始游戏
		自玩时，使用最新权重初始化网络播放器
		"""
		if not self.auto_update:
			return
		latest_model_name = get_latest_weight_path()
		model_path = os.path.join(self.distributed_dir, latest_model_name)
		if self.network_w is None or self.network_b is None:
			network = resnet.get_model(
				model_path,
				create_uci_labels(),
				gpu_core=[self.gpu_num],
				filters=conf.TrainingConfig.network_filters,
				num_res_layers=conf.TrainingConfig.network_layers
			)
			self.network_w = network
			self.network_b = network
			self.model_name_in_use = model_path
		else:
			if model_path != self.model_name_in_use:
				(sess, graph), ((_, _), (_, _)) = self.network_w
				with graph.as_default():
					saver = tf.train.Saver(var_list=tf.global_variables())
					saver.restore(sess, model_path)
				self.model_name_in_use = model_path

	def end_of_game(self, cbf_name, moves, cb_file, training_dt, epoch):
		"""
		游戏结束
		"""
		trained_games = len(os.listdir(conf.ResourceConfig.distributed_datadir))
		logging.info('------------------epoch {}: trained {} games, this game used {}s'.format(
			epoch,
			trained_games,
			round(training_dt, 6),
		))


class ValidationGames(ContinousNetworkPlayGames):
	pass