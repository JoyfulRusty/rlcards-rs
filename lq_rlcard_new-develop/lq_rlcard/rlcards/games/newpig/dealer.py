# -*- coding: utf-8 -*-

import random

from rlcards.games.newpig.utils import init_deck
from rlcards.const.pig.const import FIRST_PLAY, ALL_POKER, ALL_ROLE

class GzDealer:
	"""
	供猪发牌器
	"""
	def __init__(self, np_random):
		"""
		初始化供猪发牌器属性参数
		"""
		self.np_random = np_random
		self.deck = init_deck()
		self.table = []
		self.left_count = len(self.deck)
		self.shuffle()
		self.played_cards = {role: [] for role in ALL_ROLE}

	def shuffle(self):
		"""
		todo: 洗牌
		"""
		self.np_random.shuffle(self.deck)

	def set_head_card(self):
		"""
		计算出首出卡牌[407]，并从牌盒中删除
		"""
		for card in self.deck:
			if card.card_value == FIRST_PLAY:
				self.deck.remove(card)
				return card.card_value

	def deal_cards(self, player, card_nums):
		"""
		todo: 发牌
		"""
		tmp_hand_cards = []
		# 将0号玩家固定为首出玩家
		if player.player_id == 0:
			tmp_hand_cards.append(self.set_head_card())
			self.left_count -= 1
			card_nums -= 1
		# 发牌
		for _ in range(card_nums):
			self.left_count -= 1
			card = self.deck.pop()
			tmp_hand_cards.append(card.card_value)
		tmp_hand_cards.sort()
		player.set_hand_cards(tmp_hand_cards)