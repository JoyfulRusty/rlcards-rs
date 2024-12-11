# -*- coding: utf-8 -*-

import os

from config.conf import ResourceConfig, TrainingConfig


def get_latest_weight_path():
	"""
	latest_weight路径，绝对路径
	"""
	weight_list = os.listdir(ResourceConfig.model_dir)
	weight_list = [i[:-6] for i in weight_list if '.index' in i]
	if len(weightlist) == 0:
		raise Exception('no pre_trained weights')
	latest_net_name = sorted(weight_list)[-1]
	return os.path.join(ResourceConfig.model_dir, latest_net_name)

def cycle_lr(step):
	"""
	循环学习率
	"""
	index = (step // 500) % len(TrainingConfig.lr)
	return TrainingConfig.lr[index]