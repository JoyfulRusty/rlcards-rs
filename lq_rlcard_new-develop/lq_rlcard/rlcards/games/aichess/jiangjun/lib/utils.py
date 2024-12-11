# -*- coding: utf-8 -*-

import os
import time
import json
import shutil
import logging
import argparse

from config import conf

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

project_path = os.path.abspath(os.path.dirname(__file__))


def get_sorted_weight_list():
	"""
	排序权重大小
	"""
	weight_list = os.listdir(conf.ResourceConfig.model_dir)
	weight_list = [i[:-6] for i in weight_list if '.index' in i]
	if len(weight_list) == 0:
		raise Exception('no pretrained weights')
	weight_list.sort()
	return weight_list

def get_model_pool_list():
	"""
	获取模型池列表
	"""
	model_stamps = get_sorted_weight_list()
	if len(model_stamps) <= conf.EvaluateConfig.model_pool_size + 1:
		model_pool_list = model_stamps[:-1]
	else:
		previous_model_stamp = get_sorted_weight_list()[-2]
		file_path = os.path.join(
			conf.ResourceConfig.nash_battle_local_dir,
			previous_model_stamp,
			conf.ResourceConfig.model_pool_list_json
		)
		# 判断文件路径是否存在
		if os.path.exists(file_path):
			with open(file_path, 'r') as f:
				model_pool_dict = json.load(f)
			model_pool_list = model_pool_dict['model_pool']
		else:
			model_pool_list = model_stamps[(-1 * conf.EvaluateConfig.model_pool_size - 1): -1]

	return model_pool_list

def get_latest_weight_path():
	"""
	获取最后的权重路径
	"""
	return get_sorted_weight_list()[-1]

def check_if_dir_exit(dir_name):
	"""
	检查文件
	"""
	if not os.path.exists(dir_name):
		try:
			os.makedirs(dir_name)
		except ValueError:
			logging.error('creating {} error'.format(dir_name))

def get_train_url(user_train_url=None):
	"""
	获取训练url
	"""
	if user_train_url:
		if 'MA_OUTPUTS' not in os.environ:
			raise ValueError('MA_OUTPUTS not found')
		train_url = None
		ma_outputs = json.loads(os.environ['MA_OUTPUTS'])
		for output in ma_outputs['outputs']:
			if output['parameter']['label'] == 'train_url':
				train_url = 's3:/' + output['data_source']['obs']['obs_url']
				break
		if train_url is None:
			raise ValueError('train_url not found. MA_OUTPUTS=%s' % os.environ['MA_OUTPUTS'])
		return train_url
	else:
		return None

def str2bool(v):
	"""
	字符转化池
	"""
	if v.lower() in ('true', '1'):
		return True
	elif v.lower() in ('false', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')

def str2lower(v):
	"""
	字符转换为小写
	"""
	return v.lower()

def add_score(one_dic, key, point):
	"""
	添加分数
	"""
	one_dic.setdefault(key, 0)
	one_dic[key] += point

def cal_points(game_plays):
	"""
	计算位置点
	"""
	point_dic = {}
	for one_game in game_plays:
		if oneg_ame[-3:] != 'cbf':
			continue
		winner = one_game.split('_')[-1].split('.')[0]
		player1 = one_game.split('_')[-2].split('-')[0]
		player2 = one_game.split('_')[-2].split('-')[1]
		assert(winner in ['w', 'b', 'peace'])
		if winner == 'w':
			add_score(point_dic, player1, 1)
			add_score(point_dic, player2, 0)
		elif winner == 'b':
			add_score(point_dic, player1, 0)
			add_score(point_dic, player2, 1)
		else:
			pass
	return point_dic

def cal_res():
	"""
	计算剩余
	"""
	# 获取最后权重路径
	new_name = get_latest_weight_path()
	record_dir = '{}/{}'.format(conf.ResourceConfig.validate_dir, new_name)
	game_plays = os.listdir(record_dir)
	point_dic = cal_points(game_plays)
	game_num = len(game_plays)
	if game_num > 0:
		wins = point_dic.get('new_net', 0)
		loses = point_dic.get('old_net', 0)
		peaces = game_num - wins - loses
		logging.info('win rate : {}, peace rate: {}'.format(round(wins / game_num, 2), round(peaces / game_num, 2)))
	else:
		logging.info('no validate games exists!!!')

def get_old_files_list():
	"""
	获取老文件列表
	"""
	weights_dir = os.path.join(project_path, 'data/prepare_weight')
	weights_old = os.listdir(weights_dir)
	validate_dir = os.path.join(project_path, 'data/validate')
	validate_dirs_old = os.listdir(validate_dir)
	history_self_play_dir = os.path.join(project_path, 'data/history_self_plays')
	history_self_play_old = os.listdir(history_self_play_dir)
	return weights_old, validate_dirs_old, history_self_play_old

def upload_dir(dirs_old, local_dir, output_dir):
	"""
	上传文件
	"""
	check_if_dir_exit(output_dir)
	dirs_now = os.listdir(local_dir)
	for dir_data in dirs_now:
		if dir_data not in dirs_old:
			dir_ori = os.path.join(local_dir, dir_data)
			dir_dst = os.path.join(output_dir, dir_data)
			logging.info('upload {}'.format(dir_data))
			shutil.copytree(dir_ori, dir_dst)
			time.sleep(1)
			if os.path.exists(dir_ori):
				try:
					shutil.rmtree(dir_ori)
				except FileNotFoundError:
					pass

def upload_final_weight(args):
	"""
	上传最终权重
	"""
	output_final_weight_path = os.path.join(args.train_url, 'final_weights')
	if args.name:
		output_final_weight_path = os.path.join(args.train_url, '{}_final_weights'.format(args.name))
	check_if_dir_exit(output_final_weight_path)
	final_weight_name = get_latest_weight_path()
	for f in ['data-00000-of-00001', 'meta', 'index']:
		src = os.path.join(conf.ResourceConfig.model_dir, '{}.{}'.format(final_weight_name, f))
		dst = os.path.join(output_final_weight_path, '{}.{}'.format(final_weight_name, f))
		if os.path.exists(src):
			shutil.copyfile(src, dst)
		else:
			logging.error('{} not exist'.format(final_weight_name))
			break
		logging.info('saving {}'.format(final_weight_name))
	json_file_path = os.path.join(output_final_weight_path, 'model_info.json')
	model_info_dict = {
		"type": "tf",
		"model_name": final_weight_name
	}
	with open(json_file_path, 'w+') as f:
		json.dump(model_info_dict, f)

def upload_training_weights(args, weights_old):
	"""
	上传训练权重
	"""
	output_weight_path = os.path.join(args.train_url, 'training_res')
	check_if_dir_exit(output_weight_path)
	weights_dir = os.path.join(project_path, 'data/prepare_weight')
	weights_now = os.listdir(weights_dir)
	for weight in weights_now:
		if weight not in weights_old:
			weight_ori = os.path.join(weights_dir, weight)
			weight_dst = os.path.join(output_weight_path, weight)
			logging.info('saving {}'.format(weight))
			shutil.copyfile(weight_ori, weight_dst)
			time.sleep(1)
			if os.path.exists(weight_ori):
				try:
					os.remove(weight_ori)
				except FileNotFoundError:
					pass

def upload_files(args, weights_old, validate_dirs_old, history_self_play_old):
	"""
	上传文件
	"""
	validate_dir = os.path.join(project_path, 'data/validate')
	history_self_play_dir = os.path.join(project_path, 'data/history_self_plays')

	# upload validate
	upload_dir(validate_dirs_old, validate_dir, os.path.join(args.train_url, 'validate'))

	# upload history_self_play
	upload_dir(history_self_play_old, history_self_play_dir, os.path.join(args.train_url, 'history_self_play'))

def restore_pretrained_model(args):
	"""
	恢复预训练模型
	"""
	weights_files = os.listdir(args.data_url)
	restore_weight_list = [i[:-6] for i in weights_files if '.index' in i]
	latest_weight_name = sorted(restore_weight_list)[-1]
	for f in ['data-00000-of-00001', 'meta', 'index']:
		if '{}.{}'.format(latest_weight_name, f) not in weights_files:
			logging.info('{}.{} not exists, using init weights for training'.format(latest_weight_name, f))
			return
	stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
	for f in ['data-00000-of-00001', 'meta', 'index']:
		src = os.path.join(args.data_url, '{}.{}'.format(latest_weight_name, f))
		dst = os.path.join(conf.ResourceConfig.model_dir, '{}.{}'.format(stamp, f))
		shutil.copyfile(src, dst)
	logging.info('using {} for training'.format(latest_weight_name))

def create_yun_dao_dir(dir_name):
	"""
	创建yun dao文件
	"""
	if not mox.file.exists(dir_name):
		mox.file.make_dirs(dir_name)


def init_dir():
	"""
	初始化文件夹
	"""
	check_if_dir_exit(conf.ResourceConfig.distributed_datadir)
	check_if_dir_exit(conf.ResourceConfig.model_dir)
	check_if_dir_exit(conf.ResourceConfig.nash_battle_local_dir)
	check_if_dir_exit(conf.ResourceConfig.history_selfplay_dir)
	check_if_dir_exit(conf.ResourceConfig.validate_dir)
	check_if_dir_exit(conf.ResourceConfig.tensorboard_dir)

	create_yun_dao_dir(conf.ResourceConfig.new_data_yundao_dir)
	create_yun_dao_dir(conf.ResourceConfig.validate_yundao_dir)
	create_yun_dao_dir(conf.ResourceConfig.pool_weights_yundao_dir)
	create_yun_dao_dir(conf.ResourceConfig.nash_weights_yundao_dir)
	create_yun_dao_dir(conf.ResourceConfig.nash_battle_yundao_dir)
	create_yun_dao_dir(conf.ResourceConfig.nash_battle_yundao_share_dir)
	create_yun_dao_dir(conf.ResourceConfig.tensorboard_yundao_dir)

def sorted_custom(input_list):
	"""
	排序
	"""
	input_list = [int(_) for _ in input_list]
	input_list.sort()
	input_list = [str(_) for _ in input_list]
	return input_list