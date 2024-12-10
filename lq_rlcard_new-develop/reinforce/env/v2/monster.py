# -*- coding: utf-8 -*-

import torch

from reinforce.games.monster_v2.onehot import OneHot
from reinforce.games.monster_v2.game import MonsterGame


class MonsterEnv:
	"""
	打妖怪环境
	"""

	def __init__(self, env, device="cpu"):
		"""
		初始化环境参数
		"""
		self.name = env
		self.game = MonsterGame()
		self.device = 'cuda:' + str(device) if device != "cpu" else "cpu"
		self.positions = ["down", "right", "up", "left"]

	def format_obs(self, obs):
		"""
		将观测数据转换为神经网络输入格式
		"""
		position = obs["position"]
		x_batch = torch.from_numpy(obs['x_batch']).to(torch.device(self.device))
		z_batch = torch.from_numpy(obs['z_batch']).to(torch.device(self.device))
		x_no_action = torch.from_numpy(obs['x_no_action']).to(torch.device(self.device))
		z = torch.from_numpy(obs['z']).to(torch.device(self.device))
		obs = {
			'x_batch': x_batch,
			'z_batch': z_batch,
			'legal_actions': obs['legal_actions'],
			"last_action": obs['last_action']
		}
		return position, obs, x_no_action, z

	def initial(self):
		"""
		初始化游戏环境
		"""
		obs = self.game.reset()
		position, obs, x_no_action, z = self.format_obs(obs)
		return position, obs, dict(obs_x_no_action=x_no_action, obs_z=z)

	def step(self, action):
		"""
		迭代动作
		"""
		# 迭代下一个合法动作
		obs, episode_return, done = self.game.step(action)
		if done:
			# 初始化游戏环境
			obs = self.game.reset()
		# 转换为神经网络输入格式
		position, obs, x_no_action, z = self.format_obs(obs)
		return position, obs, dict(done=done, episode_return=episode_return, obs_x_no_action=x_no_action, obs_z=z)

	@staticmethod
	def actions2tensor(cards):
		"""
		将卡牌转换为张量
		"""
		matrix = OneHot.action2tensor(cards)
		matrix = torch.from_numpy(matrix)
		return matrix