# -*- coding: utf-8 -*-

import os
import random
import time
import logging

from config.conf import TrainingConfig, ResourceConfig


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


class Sampler:
	"""
	采样
	"""
	def __init__(self):
		"""
		初始化参数
		"""
		self.distributed_data_dir = ResourceConfig.distributed_data_dir
		self.all_games = TrainingConfig.sample_games

	@staticmethod
	def is_full(directory):
		"""
		判断是否满[5000个文件]
		"""
		files = os.listdir(directory)
		return len(files) > ResourceConfig.block_min_games - 1

	def sample(self):
		"""
		返回采样的游戏路径
		"""
		# 采样数据
		block_dirs = os.listdir(self.distributed_data_dir)
		# 等待足够的数据进行训练
		data_flag = True
		while data_flag:
			block_dirs = os.listdir(self.distributed_data_dir)
			if len(block_dirs) < ResourceConfig.train_min_block:
				logging.info('waiting for self_play data')
				time.sleep(60)
			else:
				data_flag = False
		block_dirs = [int(_) for _ in block_dirs]
		block_dirs = sorted(block_dirs, reverse=True)
		block_dirs = [str(_) for _ in block_dirs]
		if not self.is_full(os.path.join(self.distributed_data_dir, block_dirs[0])):
			block_dirs = block_dirs[1:]
		if len(block_dirs) > ResourceConfig.train_max_block:
			# 返回新模型
			block_dirs = block_dirs[:ResourceConfig.train_max_block]
		blocks_num = len(block_dirs)
		games_pre_block = int(self.all_games / blocks_num)
		file_list = []
		for i in range(blocks_num):
			block_dir = os.path.join(self.distributed_data_dir, block_dirs[i])
			block_files = os.listdir(block_dir)
			selected_files = random.sample(block_files, games_pre_block)
			file_list += [os.path.join(block_dir, i) for i in selected_files]
		return file_list