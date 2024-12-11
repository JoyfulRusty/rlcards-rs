# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np

from rlcards.const.monster import const


class MonsterJudge:
	"""
	判别器
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

		if over_flag == 4:
			return True

		return False

	def judge_payoffs(self, game):
		"""
		计算对局结束后奖励
		"""
		payoffs = np.array(const.INIT_REWARDS, np.float32)

		payoffs = self.calc_each_player_reward(game, payoffs)
		for player in game.players:
			payoffs[player.player_id] += player.pick_rewards

		# print("############玩家奖励收益: ", payoffs)

		return payoffs

	@staticmethod
	def calc_pick_operate_count(traces):
		"""
		统计本次对局中是否进行过捡牌
		"""
		trace_history = [action[1] for action in traces]
		if not Counter(trace_history).get('PICK_CARDS', 0):
			return True
		return False

	def calc_each_player_reward(self, game, payoffs):
		"""
		根据玩家分数，计算每一位玩家的奖励
		"""
		# 基础奖励
		# reward = 0.2

		# 是否捡过牌
		no_pick_actions = self.calc_pick_operate_count(game.round.traces)
		if no_pick_actions:
			for ps in game.pid_to_golds:
				payoffs[ps[0]] = 0

		# 存在捡牌动作
		else:
			for ps in game.pid_to_golds:
				if ps[1] < 0:
					# payoffs[ps[0]] = (ps[1] * 1e-3 * reward + (-reward))
					payoffs[ps[0]] = ps[1] / pow(10, 4)
				else:
					# payoffs[ps[0]] = (ps[1] * 1e-3 * reward)
					# reward -= 0.05
					payoffs[ps[0]] = ps[1] / pow(10, 4)

		return payoffs