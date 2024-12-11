# -*- coding: utf-8 -*-

class CFG(object):
	"""
	todo: 表示通过应用程序使用的静态配置文件

	num_iterrations：迭代次数。
	num_games：每次迭代中玩的自玩游戏的数量。
	num_mcts_sims：每场比赛的mcts模拟次数。
	c_puct：MCTS中使用的探索级别。
	l2_val：训练期间使用的l2权重正则化水平。
	动量：动量优化器的动量参数。
	learning_rate：动量优化器的学习速率。
	t_policy_val：策略预测的值。
	temp_init：用于控制勘探的初始温度参数。
	temp_final：控制勘探的最终温度参数。
	temp_thresh：温度初始值变为最终值的阈值。
	历元数：训练期间的历元数。
	batch_size：训练的批量大小。
	dirichlet _alpha：dirichlet噪波的alpha值。
	epsilon：用于计算狄利克雷噪声的epsilon值。
	model_directory：存储模型的目录的名称。
	num_eval_games：要进行评估的自玩游戏的数量。
	eval_win_rate：获胜率需要
	"""
	num_iterations = 4
	num_games = 30
	num_mcts_sims = 30
	c_puct = 1
	l2_val = 0.0001
	momentum = 0.9
	learning_rate = 0.01
	t_policy_val = 0.0001
	temp_init = 1
	temp_final = 0.001
	temp_thresh = 10
	epochs = 10
	batch_size = 128
	dirichlet_alpha = 0.5
	epsilon = 0.25
	model_directory = "./connect_four/models/"
	num_eval_games = 12
	eval_win_rate = 0.55
	load_model = 1
	human_play = 0
	resnet_blocks = 5
	record_loss = 1
	loss_file = "loss.txt"
	game = 2