# -*- coding: utf-8 -*-

import os
import argparse

from rlcards.games.pig.dmc.dmc import train

if __name__ == '__main__':
	# 添加解析参数
	parser = argparse.ArgumentParser(description='Pytorch Training Games AI')

	# 常规设置
	parser.add_argument(
		'--x_pid',
		default='pig',
		help='experiment id (default: pig)'
	)

	parser.add_argument(
		'--save_interval',
		default=180,
		type=int,
		help='time interval (int minutes) as which to save models'
	)

	parser.add_argument(
		'--objective',
		default='adp',
		type=str,
		choices=['adp', 'wp', 'log_adp'],
		help='use adp or wp as reward (default: adp)'
	)

	# 训练设置
	parser.add_argument(
		'--actor_device_cpu',
		default=True,
		action='store_true',
		help='use cpu as actor device'
	)

	parser.add_argument(
		'--gpu_devices',
		default='0',
		type=str,
		help='which gpus to be use for training'
	)

	parser.add_argument(
		'--num_actor_devices',
		default=1,
		type=int,
		help='the number of devices used for simulation'
	)

	parser.add_argument(
		'--num_actors',
		default=1,
		type=int,
		help='the number of actors for each simulation device'
	)

	parser.add_argument(
		'--training_device',
		default='cpu',
		type=str,
		help='the index of the gpu used for training models, cpu meaning using cpu'
	)

	parser.add_argument(
		'--load_model',
		action='store_true',
		default=True,
		help='load an existing model'
	)

	parser.add_argument(
		'--disable_checkpoint',
		action='store_true',
		help='disable saving checkpoint'
	)

	parser.add_argument(
		'--save_dir',
		default='results/pig/dmc_result',
		help='root dir where experiments data will be saved'
	)

	# 超参数
	parser.add_argument(
		'--total_frames',
		default=1000000000,
		type=int,
		help='total environment frames to dmc for'
	)

	parser.add_argument(
		'--exp_epsilon',
		default=0.01,
		type=float,
		help='the probability for exploration'
	)

	parser.add_argument(
		'--batch_size',
		default=32,
		type=int,
		help='learner batch size'
	)

	parser.add_argument(
		'--num_buffers',
		default=50,
		type=int,
		help='number of shared-memory buffers'
	)

	parser.add_argument(
		'--unroll_length',
		default=100,
		type=int,
		help='the unroll length time dimension'
	)

	parser.add_argument(
		'--num_threads',
		default=4,
		type=int,
		help='number learner threads'
	)

	parser.add_argument(
		'--max_grad_norm',
		default=40.,
		type=float,
		help='max norm of gradients'
	)

	# 优化器设置
	parser.add_argument(
		'--learning_rate',
		default=0.0001,
		type=float,
		help='learning tate'
	)

	parser.add_argument(
		'--alpha',
		default=0.99,
		type=float,
		help='RMSProp smoothing constant'
	)

	parser.add_argument(
		'--momentum',
		default=0,
		type=float,
		help='RMSProp momentum'
	)

	parser.add_argument(
		'--epsilon',
		default=1e-5,
		type=float,
		help='RMSProp epsilon'
	)

	flags = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
	train(flags)