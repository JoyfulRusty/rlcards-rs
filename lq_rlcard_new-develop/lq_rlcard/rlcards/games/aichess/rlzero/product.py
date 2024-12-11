# -*- coding: utf-8 -*-

import os
import time
import uuid
import queue
import pickle
import datetime
import requests

from threading import Thread
from collections import deque
from rlcards.games.aichess.rlzero.env import memory
from rlcards.games.aichess.rlzero.config import CONFIG
from rlcards.games.aichess.rlzero.env.game import Game
from rlcards.games.aichess.rlzero.env.board import Board
from rlcards.games.aichess.rlzero.util.log import debug_log
from rlcards.games.aichess.rlzero.model import PolicyValueNet
from rlcards.games.aichess.rlzero.env.player import MCTSPlayer
from rlcards.games.aichess.rlzero.env.ucci import get_cep_move_func


# todo: 与象棋引擎[cep]对战生成对局棋谱数据，用于模型训练

class CollectPipeline:
	"""
	收集管道
	"""
	def __init__(self):
		"""
		初始化管道参数
		"""
		self.c_p_uct = CONFIG['c_p_uct']
		self.use_cep = CONFIG['use_cep']
		self.play_out = CONFIG['play_out']
		self.n_play_out = CONFIG['play_out']
		self.train_flag = CONFIG['train_flag']
		self.buffer_size = CONFIG['buffer_size']
		self.play_with_cep = CONFIG['play_with_cep']
		self.model_version = CONFIG['model_version']
		self.one_piece_path = CONFIG['one_piece_path']
		self.curr_model_path = CONFIG['curr_model_path']
		self.save_buffer_path = CONFIG['train_data_path']
		self.model_version_path = CONFIG['model_version_path']

		self.temp = 1
		self.mct_player = None
		self.data_buffer = deque(maxlen=self.buffer_size)
		self.policy_value_net = PolicyValueNet(model_file=self.curr_model_path)

	def load_model(self, input_dict):
		"""
		加载模型
		"""
		return_dict = input_dict[0]
		# 读取模型版本
		self.model_version = return_dict.get("model_version", open(self.model_version_path, 'r').read())
		# 保存缓存路径
		self.save_buffer_path = f"{CONFIG['one_piece_path']}/{self.model_version}"
		# 判断当前是否存在缓存路径，不存在则创建此路径
		if not os.path.exists(self.save_buffer_path):
			os.makedirs(self.save_buffer_path)
		# 判断是否存在当前模型路径，则不存则创建此路径
		if not os.path.exists(self.curr_model_path):
			model_path = self.curr_model_path
		else:
			model_path = None
		# 构建策略网络
		self.policy_value_net = PolicyValueNet(model_file=model_path)
		# 构建蒙特卡洛搜索树玩家对象
		self.mct_player = MCTSPlayer(
			self.policy_value_net.policy_value_fn,
			c_p_uct=self.c_p_uct,
			n_play_out=self.n_play_out,
			is_self_play=1
		)
		return_dict['policy_value_net'] = self.policy_value_net

	def load_model_by_best_new(self, input_dict):
		"""
		加载最新模型
		"""
		self.model_version = open(self.model_version_path, 'r').read()
		file_name = self.curr_model_path.split('/')[-1]
		data = {
			'file_name': file_name,
			'model_version': self.model_version
		}
		model_dir = data.get('model_dir', self.model_version)
		self.load_model(input_dict)
		return model_dir

	def save_buffer(self, model_version, file_name, data_dict):
		"""
		保存对局棋盘buffer
		"""
		# 加载读取棋谱缓存路径
		self.save_buffer_path = f"{CONFIG['one_piece_path']}/{model_version}"
		# 判断当前棋谱缓存路径是否存在，不存在则创建
		if not os.path.exists(self.save_buffer_path):
			os.makedirs(self.save_buffer_path)
		# 读取路径并将数据写入保存到.pkl中
		real_path = f'{self.save_buffer_path}/{file_name}'
		with open(real_path, 'wb') as f:
			pickle.dump(data_dict, f)
			print("以.pkl文件形式保存当前对局棋谱数据")

	def check_train(self):
		"""
		检查是否处于训练状态
		"""
		if os.path.exists(self.train_flag):
			with open(self.train_flag, 'r') as f:
				res = f.read()
			if res == 'start':
				print('训练中，暂停')
				time.sleep(60 * 10)
				return self.check_train()
		return True

	@staticmethod
	def get_eq_ui_data(play_data):
		"""
		获取eq数据
		"""
		extend_data = []
		# shape is [9, 10, 9], 走子概率，赢家
		for state, mct_s_prob, winner in play_data:
			extend_data.append(memory.zip_state_mct_s_prob((state, mct_s_prob, winner)))
		return extend_data

	def collect_self_play_data(self, process_id, game, input_dict):
		"""
		收集自我对弈数据
		"""
		return_dict = input_dict[0]
		self.check_train()
		model_dir = self.load_model_by_best_new(input_dict)
		model_pkl = self.curr_model_path.split('/')[-1]
		policy_value_net = return_dict.get('policy_value_net', None)
		if policy_value_net:
			print(f'已获取最新模型: {model_pkl}')
			self.policy_value_net = policy_value_net

		if self.use_cep:
			# print("使用象棋引擎CEP对战!")
			policy_func = get_cep_move_func
		else:
			# print("使用训练模型对战!")
			policy_func = self.policy_value_net.policy_value_fn

		# print("是否使用CEP: ", self.play_with_cep)
		if self.play_with_cep:
			is_self_play = 0
		else:
			is_self_play = 1

		mct_player = MCTSPlayer(
			policy_func,
			c_p_uct=self.c_p_uct,
			n_play_out=self.n_play_out,
			is_self_play=is_self_play
		)
		winner, play_data, moves, log_info = game.start_self_play(
			mct_player,
			temp=self.temp,
			is_shown=True,
			cep_pos=process_id % 2
		)
		play_data = list(play_data)[:]
		play_data = self.get_eq_ui_data(play_data)[:]
		now_time = str(datetime.datetime.today()).split('.')[0]
		file_uid = uuid.uuid1()
		data_dict = {
			'play_data': play_data,
			'time': now_time,
			'moves': moves,
			'winner': winner,
			'log_info': log_info,
			'file_uid': file_uid,
			'user_id': CONFIG['user_id'],
			'hostname': CONFIG['hostname'],
			'model_version': model_dir,
			'model_path': CONFIG['curr_model_path']
		}
		file_name = f'{file_uid}.pkl'
		debug_log.debug(log_info + f' file_name:{file_name}, winner:{winner}, date: {now_time.split()[0]}')
		# 缓存[cep vs ai]对局棋谱，用于模型训练
		self.save_buffer(model_dir, file_name, data_dict)

	def run(self, process_id, return_dict):
		"""
		运行
		"""
		start_time = time.time()
		board = Board()
		game = Game(board)
		self.collect_self_play_data(process_id, game, return_dict)
		end_time = time.time()
		print(f"本次对局结束，耗时: {end_time - start_time}s, 开始新对局")

def main(i, collection_pipeline, input_list):
	"""
	主流程
	"""
	collection_pipeline.run(i, input_list)


class Store(Thread):
	def __init__(self, store, queue):
		Thread.__init__(self)
		self.queue = queue
		self.store = store

	def run(self):
		try:
			main(self.store[0], self.store[1], self.store[2])
		except Exception as e:
			print(e)
		finally:
			self.queue.get()
			self.queue.task_done()

def start_thread():
	collecting_pipeline = CollectPipeline()
	q = queue.Queue(CONFIG['thread_nums'])
	input_list = [{}]
	for i in range(1):
		q.put(i)
		t = Store((i, collecting_pipeline, input_list), q)
		t.start()
	q.join()
	print('over')

def start_one():
	collecting_pipeline = CollectPipeline()
	input_list = [{}]
	for i in range(1000):
		main(i, collecting_pipeline, input_list)
	print('over')

if __name__ == '__main__':
	start_thread()