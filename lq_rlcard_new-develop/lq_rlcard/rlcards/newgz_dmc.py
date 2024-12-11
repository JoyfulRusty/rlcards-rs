# -*- coding: utf-8 -*-


import os
import argparse

from rlcards.envs import registration
from rlcards.agents.dmc_agent.dmc.gz import DMCTrainer

def dmc_train(flags):
	# 创建环境
	env = registration.make(flags.env)
	# 初始化dmc训练模型
	trainer = DMCTrainer(
		env,
		cuda=flags.cuda,  # cuda设备
		load_model=flags.load_model,  # 是否加载现有模型
		x_pid=flags.x_pid,  # 编号
		save_dir=flags.save_dir,  # 保存文件夹
		save_interval=flags.save_interval,  # 多少批次保存一次
		num_actor_devices=flags.num_actor_devices,  # 用于模拟设备的设备数量
		num_actors=flags.num_actors,  # 每个模拟设备的参与者数量
		training_device=flags.training_device  # 用于训练模型的GPU索引或CPU
	)
	# 训练 monster dmc agents
	trainer.start()


if __name__ == '__main__':
	parser = argparse.ArgumentParser("DMC Pig in RLCard")
	parser.add_argument(
		'--env',
		type=str,
		default='pig',
		choices=[
			'mahjong_bak',
			'monster'
			'pig'
		])

	parser.add_argument(
		'--cuda',
		type=str,
		default="")

	parser.add_argument(
		'--load_model',
		action='store_true',  # 存储真实
		default=-True,
		help='加载现有模型')

	parser.add_argument(
		'--x_pid',
		default='pig',
		help='训练编号(default: pig)')

	parser.add_argument(
		'--save_dir',
		default='results/dmc/pig_conv',
		help='保存训练数据的根目录')

	parser.add_argument(
		'--save_interval',
		default=120,
		type=int,
		help='保存模型的时间间隔(分钟)')

	parser.add_argument(
		'--num_actor_devices',
		default=3,
		type=int,
		help='用于模拟的设备数量')

	parser.add_argument(
		'--num_actors',
		default=3,
		type=int,
		help='每个模拟设备的参与者数量')

	parser.add_argument(
		'--training_device',
		default='0',
		type=str,
		help='用于训练模型的GPU的索引')

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
	dmc_train(args)