# -*- coding: utf-8 -*-

import random
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
		return True if round.winner_id is not None else False

	def judge_payoffs(self, game):
		"""
		奖励
		"""
		# 初始化奖励值
		payoffs = np.array([0.0, 0.0], dtype=np.float32)

		# 计算奖励
		for player in game.players:
			if game.round.winner_id == player.player_id:
				payoffs[player.player_id] += self.calc_cards_by_rewards(player)
			payoffs[player.player_id] += player.action_reward
			payoffs[player.player_id] += player.extra_reward
			payoffs[player.player_id] += player.cards_reward

		# print("输出玩家对应奖励: {}".format(payoffs))

		return payoffs

	@staticmethod
	def calc_cards_by_rewards(player):
		"""
		根据卡牌好坏再添加额外奖励
		"""
		cards = [card % 100 for card in player.curr_hand_cards]
		res_card = player.combine_cards[-1] % 100 / 100
		res_rewards = random.uniform(res_card/2, res_card)
		# 水鱼天下
		if len(set(cards)) == 1:
			return random.uniform(0.5, 0.75) + res_rewards
		# 水鱼
		elif len(set(cards)) == 2:
			random.uniform(0.25, 0.5) + res_rewards
		# 正常牌型
		return random.uniform(0.15, 0.35) + res_rewards