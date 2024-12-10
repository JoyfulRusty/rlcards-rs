# -*- coding: utf-8 -*-

from collections import Counter

from reinforce.const.monster.const import BASE_GOLD, CARD_VALUES


class MonsterPlayer:
	"""
	玩家
	"""

	def __init__(self, player_id, np_random):
		"""
		初始化打妖怪游戏中属性参数
		"""
		self.is_out = False
		self.is_all = False
		self.pick_rewards = 0.0
		self.curr_hand_cards = []
		self.round_pick_cards = []
		self.receive_pick_cards = []
		self.np_random = np_random
		self.player_id = player_id

	def pick_cards(self, players, picked_cards, golds, bust, state):
		"""
		捡牌
		"""
		self.round_pick_cards.extend(picked_cards)
		self.receive_pick_cards.extend(picked_cards)
		# 根据捡牌数量赋值而损失的奖励
		if len(state["actions"]) > 1:
			self.pick_rewards += len(self.round_pick_cards) / 100

		# todo: 支付金币并判断是否破产
		self.calc_pick_to_bust(players, golds, bust)

	def card_rewards(self, pick_cards):
		"""
		根据卡牌计算奖励
		"""
		for card in pick_cards:
			self.pick_rewards -= CARD_VALUES.get(card % 100, 0)

	def calc_pick_to_bust(self, players, golds, bust):
		"""
		todo: 计算玩家捡牌后，是否破产，破产玩家将退出本局游戏
		"""
		bust_flag = 0.0
		pay_pick_golds = len(self.round_pick_cards) * BASE_GOLD

		# 捡牌未破产
		if self.count_bust(bust) == 0:
			avg_picks_to_golds = float(pay_pick_golds / 3)
		# 捡牌破产
		else:
			bust_nums = self.count_bust(bust)
			if bust_nums == 3:
				avg_picks_to_golds = 0.0
			else:
				avg_picks_to_golds = float(pay_pick_golds / bust_nums)

		# 计算支付本轮捡牌后所剩余的金币，判断玩家是否破产
		pay_after_golds = golds[self.player_id] - float(pay_pick_golds)
		golds[self.player_id] = pay_after_golds
		if float(pay_after_golds) <= bust_flag:
			players[self.player_id].is_out = True

		# 更新每位玩家持有的金币数量
		for p_id, p in enumerate(players):
			if p_id == self.player_id:
				continue
			if p.is_out:
				continue
			if p.is_all:
				continue
			golds[p_id] += float(avg_picks_to_golds)

	@staticmethod
	def count_bust(bust):
		"""
		计算破产玩家数量
		"""
		count = Counter(list(bust.values()))
		return count.get(True, 0)