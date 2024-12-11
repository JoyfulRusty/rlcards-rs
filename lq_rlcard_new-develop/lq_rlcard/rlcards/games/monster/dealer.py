# -*- coding: utf-8 -*-

import functools

from rlcards.const.monster import const
from rlcards.games.monster.utils import compare2cards, init_28_deck, magic_init_4_deck


class MonsterDealer:
	"""
	发牌器
	"""
	def __init__(self, np_random):
		"""
		初始化卡牌参数
		"""
		self.np_random = np_random
		self.jd_card = None  								# 方块J -> [111]
		self.landlord_id = None                             # 方块J玩家ID
		self.deck = init_28_deck()                          # 正常卡牌(4 x 7 = 28)
		self.magic_deck = magic_init_4_deck()               # 万能牌(1 x 4 = 4)
		self.shuffle()                                      # 洗牌
		self.action_history = []  							# 出牌历史
		self.played_cards = {"down": [], 'right': [], 'up': [], 'left': []}			    # 出牌记录

	def shuffle(self):
		"""
		TODO: 洗牌(万能牌不洗)
		"""
		self.np_random.shuffle(self.deck)

	def deal_cards(self, players, nums=7):
		"""
		TODO: 发牌
		"""
		for player in players:
			for _ in range(nums):
				card = self.deck.pop()
				# 计算方块J持有玩家
				if card.card_value == const.ALL_CARDS[4]:  # [111]为方块J
					self.jd_card = [card.card_value]
					self.landlord_id = player.player_id
				player.curr_hand_cards.append(card.get_card_value())

			if self.magic_deck:
				magic_card = self.magic_deck.pop()
				player.curr_hand_cards.append(magic_card.get_card_value())

			player.curr_hand_cards.sort(key=functools.cmp_to_key(compare2cards))

		return self.landlord_id

	def play_card(self, curr_player, action):
		"""
		TODO: 删除当前出牌并记录
		"""
		action_idx = curr_player.curr_hand_cards.index(action)
		card = curr_player.curr_hand_cards.pop(action_idx)
		if curr_player.player_id == 0:
			self.played_cards['down'].append(card)
		elif curr_player.player_id == 1:
			self.played_cards['right'].append(card)
		elif curr_player.player_id == 2:
			self.played_cards['up'].append(card)
		elif curr_player.player_id == 3:
			self.played_cards['left'].append(card)