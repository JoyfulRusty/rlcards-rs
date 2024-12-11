# -*- coding: utf-8 -*-

import os


YUN_DAO_DIR_PATH = ''
CURR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_BASE_DIR = os.path.join(CURR_PATH)


class ResourceConfig:
	"""
	资源配置
	"""
	python_executor = 'python'
	block_min_games = 5000  # 每个区块游戏的最小次数
	train_max_block = 100  # 每个区块训练的最大次数
	train_min_block = 3  # 每个区块训练的最小次数
	num_process = 10
	restore_path = None
	distributed_data_dir = os.path.join(PROJECT_BASE_DIR, 'data/distributed')
	nash_battle_local_dir = os.path.join(PROJECT_BASE_DIR, 'data/nash_battle')
	history_self_play_dir = os.path.join(PROJECT_BASE_DIR, 'data/history_self_plays')
	model_dir = os.path.join(PROJECT_BASE_DIR, 'data/models')
	validate_dir = os.path.join(PROJECT_BASE_DIR, 'data/validate')
	tensorboard_dir = os.path.join(PROJECT_BASE_DIR, 'data/tensorboard')

	# YUN DAO
	new_data_yun_dao_dir = os.path.join(YUN_DAO_DIR_PATH, 'self_play')
	validate_yun_dao_dir = os.path.join(YUN_DAO_DIR_PATH, 'validate')
	pool_weights_yun_dao_dir = os.path.join(YUN_DAO_DIR_PATH, 'pool_weights')
	nash_weights_yun_dao_dir = os.path.join(YUN_DAO_DIR_PATH, 'nash_weights')
	nash_battle_yun_dao_dir = os.path.join(YUN_DAO_DIR_PATH, 'nash_battle')
	nash_battle_yun_dao_share_dir = os.path.join(YUN_DAO_DIR_PATH, 'nash_battle_share')
	tensorboard_yun_dao_dir = os.path.join(YUN_DAO_DIR_PATH, 'tensorboard')
	nash_battle_list_json = 'nash_battle_list.json'
	nash_res_json = 'nash_res.json'
	nash_res_bot_json = 'nash_res_bot.json'
	model_pool_list_json = 'model_pool_list.json'
	chosen_model_json = 'chosen_model.json'
	sub_validate_json = 'sub_validate_res.json'
	eliminated_model_json = 'eliminated_model.json'


class TrainingConfig:
	"""
	训练配置
	"""
	network_filters = 192
	network_layers = 10
	batch_size = 2048
	sample_games = 500
	c_p_uct = 1.5
	saver_step = 400
	# lr = [
	#     (0, 0.03),
	#     (10000, 0.01),
	#     (20000, 0.003),
	#     (30000, 0.001),
	#     (40000, 0.0003),
	#     (50000, 0.0001),
	# ]
	lr = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
	max_model_num = 10
	train_with_bot = True


class SelfPlayConfig:
	"""
	自玩配置
	"""
	non_cap_draw_round = 120
	train_play_out = 800
	train_temp_round = 3000
	gpu_num = 8
	num_proc_each_gpu = 4
	self_play_download_weight_dt = 30
	self_play_upload_data_dt = 60
	self_play_games_one_time = 1
	cpu_proc_num = 8
	yi_gou_cpu_proc_num = 16
	resign_score = -0.95
	game_num_to_restart = 100
	py_env = False


class EvaluateConfig:
	"""
	评估配置
	"""
	chess_play_out = 400
	val_play_out = 800
	val_temp_round = 12
	model_pool_size = 20
	nash_each_battle_num = 64
	bot_name = 'bot'
	icy_player_name = 'icy'
	nash_eva_waiting_dt = 60
	gpu_num = 8
	num_proc_each_gpu = 4
	nash_nodes_num = 2