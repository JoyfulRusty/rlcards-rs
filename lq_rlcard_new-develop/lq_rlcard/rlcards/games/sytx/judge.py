# -*- coding: utf-8 -*-

import numpy as np

class SyJudge:
	"""
	水鱼判别器
	"""
	def __init__(self, np_random):
		"""
		初始化水鱼判断参数
		"""
		self.np_random = np_random

	@staticmethod
	def judge_name(round):
		"""
		判断本局游戏是否还继续操作，作为游戏结束标识
		"""
		# 当前赢家ID不为None时，说明该局结束
		if round.winner_id is not None:
			return True
		# 本局游戏未结束，下一位玩家继续操作
		return False

	def judge_payoffs(self, game):
		"""
		奖励
		"""
		# 初始化奖励值
		payoffs = np.array([0.0, 0.0], dtype=np.float32)

		# 计算奖励
		for player in game.players:
			payoffs[player.player_id] += player.action_reward
			payoffs[player.player_id] += player.extra_reward
			payoffs[player.player_id] += player.cards_reward

		# print("输出玩家对应奖励: {}".format(payoffs))

		return payoffs