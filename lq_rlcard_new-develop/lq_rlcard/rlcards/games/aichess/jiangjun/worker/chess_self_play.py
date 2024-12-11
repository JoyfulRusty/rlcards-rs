# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from config import conf
from worker.game import DistributedSelfPlayGames

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


def self_play_gpu(gpu_num, play_times=np.inf, history_self_play_output_dir=None, epoch=0):
	"""
	自玩GPU
	"""
	cn = DistributedSelfPlayGames(
		gpu_num=gpu_num,
		n_playout=conf.SelfPlayConfig.train_play_out,
		recoard_dir=conf.ResourceConfig.distributed_data_dir,
		c_puct=conf.TrainingConfig.c_p_uct,
		distributed_dir=conf.ResourceConfig.model_dir,
		dnoise=True,
		is_selfplay=True,
		play_times=play_times,
	)
	cn.play(data_url=history_self_play_output_dir, epoch=epoch)

def self_play_cpu(play_times=np.inf, history_self_play_output_dir=None, epoch=0):
	"""
	自玩cpu
	"""
	cn = DistributedSelfPlayGames(
		gpu_num=None,
		n_playout=conf.SelfPlayConfig.train_playout,
		recoard_dir=conf.ResourceConfig.distributed_data_dir,
		c_puct=conf.TrainingConfig.c_puct,
		distributed_dir=conf.ResourceConfig.model_dir,
		dnoise=True,
		is_selfplay=True,
		play_times=play_times,
	)

	cn.play(data_url=history_self_play_output_dir, epoch=epoch)

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = "3"
	self_play_gpu(0, 200)