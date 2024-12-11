# -*- coding: utf-8 -*-

import os
import torch
import argparse

from rlcards.agents import DQNAgent
from rlcards.agents import RandomAgent
from rlcards.utils import (
	get_device,
	set_seed,
	tournament,
	reorganize,
	Logger,
	plot_curve,
)

from rlcards.envs.registration import make

def train(args):
	"""
	开始训练
	"""
	# 设置运行设备
	device = get_device()

	# 添加种子
	set_seed(args.seed)

	# 创建游戏环境
	env = make(
		args.env,
		config={
			'seed': args.seed,
		}
	)

	# 初始化训练模型代理
	# todo: DQN
	if args.algorithm == 'dqn':

		if args.load_checkpoint_path != "":
			agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
		else:
			agent = DQNAgent(
				num_actions=env.num_actions,
				state_shape=env.state_shape[0],
				mlp_layers=[64, 64],
				device=device,
				save_path=args.log_dir,
				save_every=args.save_every
			)

	# todo: n f s p
	elif args.algorithm == 'nfsp':
		from rlcards.agents import NFSPAgent
		if args.load_checkpoint_path != "":
			agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
		else:
			agent = NFSPAgent(
				num_actions=env.num_actions,
				state_shape=env.state_shape[0],
				hidden_layers_sizes=[64, 64],
				q_mlp_layers=[64, 64],
				device=device,
				save_path=args.log_dir,
				save_every=args.save_every
			)
	agents = [agent]
	for _ in range(1, env.num_players):
		agents.append(RandomAgent(num_actions=env.num_actions))
	env.set_agents(agents)

	# 开始训练
	with Logger(args.log_dir) as logger:
		for episode in range(args.num_episodes):
			# N F S P 算法
			if args.algorithm == 'nfsp':
				agents[0].sample_episode_policy()

			# 从游戏环境中获取数据
			trajectories, payoffs, _, _ = env.run(is_training=True)

			# 重新组织数据[state, action, reward, next_state, done]
			trajectories = reorganize(trajectories, payoffs)

			# 将转换馈送到代理内存中，并训练代理在这里
			# 假设DQN总是玩第一个位置，其他玩家随机玩
			for ts in trajectories[0]:
				agent.feed(ts)

			# 评估性能
			# 玩随机代理
			if episode % args.evaluate_every == 0:
				player_rewards = tournament(env,args.num_eval_games)
				for idx, reward in enumerate(player_rewards):
					logger.log_performance(
						idx,
						episode,
						reward
					)

			# 获取路径
			csv_path, fig_path = logger.csv_path, logger.fig_path

		# 绘制学习曲线
		plot_curve(csv_path, fig_path, args.algorithm)

		# 保存模型
		save_path = os.path.join(args.log_dir, 'model.pth')
		torch.save(agent, save_path)
		print('Model saved in', save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
	parser.add_argument(
		'--env',
		type=str,
		default='sytx_gz',
		choices=[
			'sytx_gz'
		],
	)
	parser.add_argument(
		'--algorithm',
		type=str,
		default='dqn',
		choices=[
			'dqn',
			'nfsp',
		],
	)
	parser.add_argument(
		'--cuda',
		type=str,
		default='',
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42,
	)
	parser.add_argument(
		'--num_episodes',
		type=int,
		default=20000000,
	)
	parser.add_argument(
		'--num_eval_games',
		type=int,
		default=20000,
	)
	parser.add_argument(
		'--evaluate_every',
		type=int,
		default=5000,
	)
	parser.add_argument(
		'--log_dir',
		type=str,
		default='results/sytx_gz_dqn/',
	)

	parser.add_argument(
		"--load_checkpoint_path",
		type=str,
		default="",
	)

	parser.add_argument(
		"--save_every",
		type=int,
		default=-1)

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
	train(args)
