# -*- coding: utf-8 -*-

import numpy as np

from rlcards.const.pig.const import INIT_REWARDS, ZOOM_SCORE_CARDS, BIG_SLAM_NUM, ALL_RED

class GzJudge:
	"""
	供猪判别器
	"""
	def __init__(self, np_random):
		"""
		初始化供猪判别器属性参数
		"""
		self.np_random = np_random

	@staticmethod
	def judge_name(players):
		"""
		判断游戏是否结束
		"""
		count = 0
		for player in players:
			if not player.curr_hand_cards:
				count += 1
			if count == 4:
				return True
		return False

	def judge_payoffs(self, game):
		"""
		todo: 计算对局结束奖励
			1.先计算分数奖励
			2.在计算分牌奖励
		"""
		payoffs = np.array(INIT_REWARDS, np.float32)
		payoffs = self.calc_score_rewards(game, payoffs)
		payoffs = self.calc_cards_rewards(game, payoffs)
		# print("############玩家奖励收益: ", payoffs)
		return payoffs

	def calc_score_rewards(self, game, payoffs):
		"""
		计算分数奖励
		"""
		for curr_p in game.players:
			payoffs[curr_p.player_id] += round((curr_p.curr_scores / 100) / 16, 2)
		return payoffs

	def calc_cards_rewards(self, game, payoffs):
		"""
		计算分牌奖励
		"""
		for curr_p in game.players:
			for sc in curr_p.receive_score_cards:
				# todo: 全红和大满贯分数加倍
				if len(curr_p.receive_score_cards) == BIG_SLAM_NUM:
					# print("#%%%%%大满贯%%%%#")
					payoffs[curr_p.player_id] += round(self.calc_big_slam_reward(), 2)
					return payoffs
				if len(curr_p.receive_score_cards) == len(ALL_RED):
					# print("#%%%%%全红%%%%#")
					payoffs[curr_p.player_id] += round(self.calc_all_hong_reward(), 2)
					return payoffs
				payoffs[curr_p.player_id] += round(ZOOM_SCORE_CARDS.get(sc, 0) / 100, 2)
		return payoffs

	def calc_big_slam_reward(self):
		"""
		计算大满贯奖励
		"""
		lv = 10
		scores = 0.0
		for sc in list(ZOOM_SCORE_CARDS.values()):
			scores += abs(sc)
		return (scores / 10) / BIG_SLAM_NUM * lv

	def calc_all_hong_reward(self):
		"""
		计算全红奖励
		"""
		lv = 10
		scores = 0.0
		for ac in ALL_RED:
			scores += abs(ZOOM_SCORE_CARDS.get(ac, 0))
		return  (scores / 10) / len(ALL_RED) * lv