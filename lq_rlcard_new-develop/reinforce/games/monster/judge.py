# -*- coding: utf-8 -*-

import random
import numpy as np

from collections import Counter
from reinforce.const.monster import const


class MonsterJudge:
	"""
	判别器类
	"""

	def __init__(self, np_random):
		"""
		初始化判别器参数
		"""
		self.np_random = np_random

	@staticmethod
	def judge_name(round_cards, players):
		"""
		判断游戏是否结束
		"""
		over_flag = 0
		if not round_cards:
			return True
		for player in players:
			if player.is_out:
				over_flag += 1
				continue
			if not player.curr_hand_cards:
				over_flag += 1
				continue
		return True if over_flag == 4 else False

	def judge_payoffs(self, game):
		"""
		计算对局结束后奖励
		"""
		payoffs = np.array(const.INIT_REWARDS, np.float32)
		payoffs = self.calc_each_player_reward(game, payoffs)
		for curr_p in game.players:
			payoffs[curr_p.player_id] -= curr_p.pick_rewards
			payoffs[curr_p.player_id] -= self.calc_pick_cards_reward(curr_p)
			payoffs[curr_p.player_id] = round(payoffs[curr_p.player_id], 3)
		return payoffs

	@staticmethod
	def calc_pick_operate_count(traces):
		"""
		统计本次对局中是否进行过捡牌
		"""
		trace_history = [action[1] for action in traces]
		pick_cards = Counter(trace_history).get('PICK_CARDS', 0)
		return False if not pick_cards else True

	def calc_each_player_reward(self, game, payoffs):
		"""
		根据玩家分数，计算每一位玩家的奖励
		"""
		pick_res = self.calc_pick_operate_count(game.round.traces)
		# 无人捡牌，则每位玩家都给1的奖励
		if not pick_res:
			for ps in game.over_round_golds:
				payoffs[ps[0]] = 1
			return payoffs
		random_rewards = 0.1
		for ps in game.over_round_golds:
			if ps[1] > 800.00:
				ps_val = ps[1] - 800.00
				val = random.uniform(0.05, random_rewards)
				payoffs[ps[0]] = ps_val / pow(10, 4) + val
			else:
				payoffs[ps[0]] = ps[1] / pow(10, 4)
		return payoffs

	def calc_pick_cards_reward(self, curr_p):
		"""
		计算捡牌获取的负奖励
		"""
		rewards = 0.0
		for card in curr_p.receive_pick_cards:
			rewards += const.CARD_VALUES.get(card % 100, 0)
		return rewards