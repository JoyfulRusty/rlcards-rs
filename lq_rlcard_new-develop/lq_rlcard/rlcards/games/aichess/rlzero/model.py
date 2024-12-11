# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast
from rlcards.games.aichess.rlzero.config import CONFIG

class ResBlock(nn.Module):
	"""
	构建残差网络
	"""
	def __init__(self, num_filters=256):
		"""
		初始化网络层参数
		"""
		super().__init__()
		# 卷积层一
		self.conv1 = nn.Conv2d(
			in_channels=num_filters,
			out_channels=num_filters,
			kernel_size=(3, 3),
			stride=(1, 1),
			padding=1
		)
		self.conv1_bn = nn.BatchNorm2d(num_filters, )
		self.conv1_act = nn.ReLU()
		# 卷积层二
		self.conv2 = nn.Conv2d(
			in_channels=num_filters,
			out_channels=num_filters,
			kernel_size=(3, 3),
			stride=(1, 1),
			padding=1
		)
		self.conv2_bn = nn.BatchNorm2d(num_filters, )
		self.conv2_act = nn.ReLU()

	def forward(self, x):
		"""
		残差网络前向传递
		"""
		y = self.conv1(x)
		y = self.conv1_bn(y)
		y = self.conv1_act(y)
		y = self.conv2(y)
		y = self.conv2_bn(y)
		y = x + y
		return self.conv2_act(y)

class Net(nn.Module):
	"""
	搭建骨干网络，输入: N, 9, 10, 9 --> N, C, H, W
	"""
	def __init__(self, num_channels=256, num_res_blocks=7):
		super().__init__()
		# 初始化特征
		self.conv_block = nn.Conv2d(
			in_channels=9,
			out_channels=num_channels,
			kernel_size=(3, 3),
			stride=(1, 1), padding=1
		)
		self.conv_block_bn = nn.BatchNorm2d(256)
		self.conv_block_act = nn.ReLU()

		# 残差块抽取特征
		self.res_blocks = nn.ModuleList(
			[ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)]
		)

		# 策略头
		self.policy_conv = nn.Conv2d(
			in_channels=num_channels,
			out_channels=16,
			kernel_size=(1, 1),
			stride=(1, 1)
		)
		self.policy_bn = nn.BatchNorm2d(16)
		self.policy_act = nn.ReLU()
		self.policy_fc = nn.Linear(16 * 9 * 10, 2086)

		# 价值头
		self.value_conv = nn.Conv2d(
			in_channels=num_channels,
			out_channels=8,
			kernel_size=(1, 1),
			stride=(1, 1)
		)
		self.value_bn = nn.BatchNorm2d(8)
		self.value_act1 = nn.ReLU()
		self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
		self.value_act2 = nn.ReLU()
		self.value_fc2 = nn.Linear(256, 1)

	def forward(self, x):
		"""
		骨干网络前向传递
		"""
		# 公共头
		x = self.conv_block(x)
		x = self.conv_block_bn(x)
		x = self.conv_block_act(x)
		for layer in self.res_blocks:
			x = layer(x)

		# 策略头
		policy = self.policy_conv(x)
		policy = self.policy_bn(policy)
		policy = self.policy_act(policy)
		policy = torch.reshape(policy, [-1, 16 * 10 * 9])
		policy = self.policy_fc(policy)
		policy = F.log_softmax(policy, dim=1)

		# 价值头
		value = self.value_conv(x)
		value = self.value_bn(value)
		value = self.value_act1(value)
		value = torch.reshape(value, [-1, 8 * 10 * 9])
		value = self.value_fc1(value)
		value = self.value_act1(value)
		value = self.value_fc2(value)
		value = torch.tanh(value)

		return policy, value

class PolicyValueNet:
	"""
	策略值网络，用来对模型进行训练
	"""
	def __init__(self, model_file=None, use_gpu=CONFIG['use_gpu'], device=CONFIG['device']):
		"""
		初始化策略值网络层参数
		"""
		self.use_gpu = use_gpu
		self.l2_const = 2e-3  # l2 正则化
		self.device = device
		self.policy_value_net = Net().to(self.device)
		self.optimizer = torch.optim.Adam(
			params=self.policy_value_net.parameters(),
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=1e-8,
			weight_decay=self.l2_const
		)

		if model_file:
			model = torch.load(model_file, map_location=torch.device(self.device))
			if self.device == 'cpu':
				model.float()
			self.policy_value_net.load_state_dict(model)  # 加载模型参数

	def policy_value(self, state_batch):
		"""
		输入一个批次的状态，输出一个批次的动作概率和状态价值
		"""
		self.policy_value_net.eval()
		state_batch = torch.tensor(state_batch).to(self.device)
		log_act_prob_s, value = self.policy_value_net(state_batch)
		log_act_prob_s, value = log_act_prob_s.cpu(), value.cpu()
		act_prob_s = np.exp(log_act_prob_s.detach().numpy())
		return act_prob_s, value.detach().numpy()

	def policy_value_fn(self, board):
		"""
		输入棋盘，返回每个合法动作的(动作，概率)元组列表，以及棋盘状态的分数
		"""
		self.policy_value_net.eval()
		# 获取合法动作列表
		legal_positions = board.legal_moves
		current_state = np.ascontiguousarray(board.get_curr_state().reshape(-1, 9, 10, 9)).astype('float16')
		current_state = torch.as_tensor(current_state).to(self.device)
		# 使用神经网络进行预测
		with autocast():  # 半精度fp16
			log_act_prob_s, value = self.policy_value_net(current_state)
		log_act_prob_s, value = log_act_prob_s.cpu(), value.cpu()
		act_prob_s = np.exp(log_act_prob_s.detach().numpy().astype('float16').flatten())
		# 只取出合法动作)
		act_prob_s = zip(legal_positions, act_prob_s[legal_positions])
		# 返回动作概率，以及状态价值
		return act_prob_s, value.detach().numpy()

	def step_policy_by_ai(self, board):
		"""
		迭代更新机器人操作
		"""
		legal_positions = board.legal_moves
		current_state = np.ascontiguousarray(board.get_curr_state().reshape(-1, 9, 10, 9)).astype('float16')
		current_state = torch.as_tensor(current_state).to(self.device)
		# 使用神经网络进行预测
		with autocast():  # 半精度fp16
			log_act_prob_s, value = self.policy_value_net(current_state)
		log_act_prob_s, value = log_act_prob_s.cpu(), value.cpu()
		act_prob_s = np.exp(log_act_prob_s.detach().numpy().astype('float16').flatten())
		max_index = np.argmax(act_prob_s[legal_positions])
		move = legal_positions[max_index]
		return move

	def save_model(self, model_file):
		"""
		保存模型
		"""
		torch.save(self.policy_value_net.state_dict(), model_file)
		torch.save(self.policy_value_net, "models/best_model.pth")

	def train_step(self, state_batch, mct_s_prob_s, winner_batch, lr=0.02):
		"""
		执行一步训练
		"""
		self.policy_value_net.train()
		# 包装变量
		state_batch = torch.tensor(state_batch).to(self.device)
		mct_s_prob_s = torch.tensor(mct_s_prob_s).to(self.device)
		winner_batch = torch.tensor(winner_batch).to(self.device)
		# 清零梯度
		self.optimizer.zero_grad()
		# 设置学习率
		for params in self.optimizer.param_groups:
			# 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
			params['lr'] = lr
		# 前向运算
		log_act_prob_s, value = self.policy_value_net(state_batch)
		value = torch.reshape(value, shape=[-1])
		# 价值损失
		value_loss = F.mse_loss(input=value, target=winner_batch)
		# 策略损失
		policy_loss = -torch.mean(torch.sum(mct_s_prob_s * log_act_prob_s, dim=1))  # 希望两个向量方向越一致越好
		# 总的损失，注意l2惩罚已经包含在优化器内部
		loss = value_loss + policy_loss
		# 反向传播及优化
		loss.backward()
		self.optimizer.step()
		# 计算策略的熵，仅用于评估模型
		with torch.no_grad():
			entropy = -torch.mean(torch.sum(torch.exp(log_act_prob_s) * log_act_prob_s, dim=1))
		return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()