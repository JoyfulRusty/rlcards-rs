# -*- coding: utf-8 -*-

import functools

from rlcards.games.doudizhu.utils import init_54_deck, cards2str, ddz_sort_card_obj

class DdzDealer:
	"""
	斗地主发牌器
	"""
	def __init__(self, np_random):
		"""
		一副有 54 张牌的牌组，包括黑色小丑和红色小丑
		:param np_random: 随机数
		"""
		self.np_random = np_random
		self.deck = init_54_deck()
		self.deck.sort(key=functools.cmp_to_key(ddz_sort_card_obj))

	def shuffle(self):
		"""
		洗牌
		:return: 将所有牌打乱
		"""
		self.np_random.shuffle(self.deck)


	def deal_cards(self, players):
		"""
		发牌
		:param players(obj): 玩家对象列表
		"""
		hand_num = (len(self.deck[:52])) // len(players)
		for index, player in enumerate(players):
			current_hand = self.deck[index * hand_num: (index + 1) * hand_num]
			current_hand.sort(key=functools.cmp_to_key(ddz_sort_card_obj))
			player.set_current_hand(current_hand)
			player.initial_hand = cards2str(player.current_hand)

	def calc_role(self, players):
		"""
		根据玩家手牌决定地主和农民
		:param players(obj): 玩家对象列表
		:return: landlord_id
		"""
		self.shuffle()
		self.deal_cards(players)
		players[0].role = 'landlord'
		self.landlord = players[0]

		players[1].role = 'peasant'
		players[2].role = 'peasant'

		self.landlord.current_hand.extend(self.deck[-3:])
		self.landlord.current_hand.sort(key=functools.cmp_to_key(ddz_sort_card_obj))
		self.landlord.initial_hand = cards2str(self.landlord.current_hand)
		return self.landlord.player_id