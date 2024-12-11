# -*- coding: utf-8 -*-

CONFIG = {
	'kill_action': 30,  # 和棋回合数
	'dirichlet': 0.2,  # 国际象棋: 0.3, 日本将棋: 0.15, 围棋: 0.03
	'play_out': 1200,  # 每次移动的模拟次数
	'cp_uct': 5,  # uct的权重
	'buffer_size': 10000,  # 经验池大小
	'policy_model_path': 'policy_model.pth',  # 策略模型路径
	'train_data_buffer_path': 'train_data_buffer.pkl',  # 数据容器路径
	'batch_size': 512,  # 每次更新的train_step数量
	'klt_arg': 0.02,  # kl散度控制
	'epochs': 5,  # 每次更新train_step的数量
	'game_batch_num': 3000,  # 训练更新的次数
	'use_frame': 'pytorch',  # paddle or pytorch
	'train_update_interval': 60,  # 模型更新间隔时间
	'use_redis': False,  # 数据存储方式，是否使用redis
	'redis_host': 'localhost',  # 本地
	'redis_port': 8888,  # 端口
	'redis_db': 0
}