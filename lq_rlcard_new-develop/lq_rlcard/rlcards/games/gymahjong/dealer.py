# -*- coding: utf-8 -*-

import random
import functools

from rlcards.games.gymahjong.utils import init_deck, compare2cards


class GyMahjongDealer:
	"""
	贵阳麻将发牌器
	"""
	def __init__(self, np_random):
		"""
		初始化贵阳麻将发牌器属性参数
		"""
		self.np_random = np_random
		self.table = []
		self.curr_card = 0
		self.left_count = 108
		self.deck = init_deck()
		self.landlord_id = None
		self.curr_player = None
		self.save_valid_operates = {}
		self.record_action_seq_history = []
		self.played_cards = [[], [], [], []]

	def shuffle(self):
		""" 洗牌 """
		self.np_random.shuffle(self.deck)

	def deal_cards(self, player, nums):
		""" 发牌 """
		for _ in range(nums):
			card = self.deck.pop()
			self.left_count -= 1
			player.curr_hand_cards.append(card.get_card_value())
		player.curr_hand_cards.sort(key=functools.cmp_to_key(compare2cards))

	def mo_cards(self, player):
		""" 摸牌 """
		curr_card = self.deck.pop()
		self.left_count -= 1
		self.curr_player = player
		self.curr_card = curr_card.get_card_value()
		player.curr_hand_cards.append(self.curr_card)
		player.curr_hand_cards.sort(key=functools.cmp_to_key(compare2cards))

	def chu_cards(self, player, card):
		""" 出牌 """
		# 从手牌中删除当前出牌并处理出牌信息
		chu_card = player.curr_hand_cards.pop(player.curr_hand_cards.index(card))
		self.table.append(chu_card)
		self.played_cards[player.player_id].append(chu_card)
		player.all_chu_cards.append((player.player_id, chu_card))

	def set_landlord_id(self, players):
		"""
		开局设庄
		"""
		player_id = [_ for _ in range(len(players))]
		self.landlord_id = sorted(
			{p_id: random.randint(1, 6) for p_id in player_id}.items(),
			key=lambda x: x[1],
			reverse=True
		)[0][0]
		return self.landlord_id

	def set_hand_cards(self, players):
		"""
		开局设置手牌
		"""
		self.shuffle()
		for player in players:
			self.deal_cards(player, 13)