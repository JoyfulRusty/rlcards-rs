# -*- coding: utf-8 -*-

from rlcards.const.pig import const

class GzPlayer:
	"""
	供猪玩家
	"""
	def __init__(self, player_id, np_random):
		"""
		初始化玩家参数
		"""
		self.player_id = player_id
		self.np_random = np_random
		self.role = ''
		self.reward = 0
		self.curr_scores = 0
		self.light_cards = []
		self.played_cards = []
		self.curr_hand_cards = []
		self.curr_score_cards = []
		self.remain_score_cards = []
		self.receive_score_cards = []

	def set_hand_cards(self, cards):
		"""
		更新玩家当前手牌
		"""
		self.curr_hand_cards.extend(cards or [])

	def set_light_cards(self, light_cards):
		"""
		设置亮牌
		"""
		self.light_cards.extend(light_cards or [])

	def calc_score_cards_by_hand(self):
		"""
		计算每一位玩家手中的分数卡牌
		"""
		for card in self.curr_hand_cards:
			if not const.SCORE_CARDS.get(card, 0):
				continue
			self.curr_score_cards.append(card)