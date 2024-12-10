# -*- coding: utf-8 -*-

import functools

from reinforce.const.monster import const
from reinforce.games.monster.player import MonsterPlayer as Player
from reinforce.games.monster.utils import compare2cards, init_28_deck, magic_init_4_deck


class MonsterDealer:
	"""
	发牌器
	"""

	def __init__(self, np_random):
		"""
		初始化卡牌参数
		"""
		self.jd_card = None
		self.landlord_id = None
		self.np_random = np_random
		self.deck = init_28_deck()
		self.magic_deck = magic_init_4_deck()
		self.shuffle()
		self.action_history = []
		self.played_cards = {"down": [], 'right': [], 'up': [], 'left': []}

	def shuffle(self):
		"""
		洗牌(万能牌不洗)
		"""
		self.np_random.shuffle(self.deck)

	def deal_cards(self, players, nums = 7):
		"""
		todo: 发牌
		"""
		for player in players:
			if self.magic_deck:
				magic_card = self.magic_deck.pop()
				player.curr_hand_cards.append(magic_card.card_value)
			for _ in range(nums):
				card = self.deck.pop()
				if card.card_value == const.ALL_CARDS[4]:
					self.jd_card = [card.card_value]
					self.landlord_id = player.player_id
				player.curr_hand_cards.append(card.card_value)
			player.curr_hand_cards.sort(key=functools.cmp_to_key(compare2cards))
		return self.landlord_id

	def play_card(self, curr_p: Player, action: int) -> None:
		"""
		todo: 删除当前出牌并记录
		"""
		action_idx = curr_p.curr_hand_cards.index(action)
		card = curr_p.curr_hand_cards.pop(action_idx)
		curr_position = {
			0: self.played_cards["down"],
			1: self.played_cards["right"],
			2: self.played_cards["up"],
			3: self.played_cards["left"]
		}.get(curr_p.player_id, 0)
		curr_position.append(card)