# -*- coding: utf-8 -*-

import functools

from rlcards.games.doudizhu.dealer import DdzDealer as Dealer
from rlcards.games.doudizhu.utils import cards2str, ddz_sort_card_obj
from rlcards.const.doudizhu.const import CARD_RANK_STR, CARD_RANK_STR_INDEX


class DdzRound:
	"""
	可以调用其他class函数来保持运行
	"""
	def __init__(self, np_random, played_cards):
		"""
		初始化round属性参数
		:param np_random: 随机数
		:param played_cards: 玩家打出的牌
		"""
		self.np_random = np_random
		self.played_cards = played_cards
		self.greater_player = None
		self.trace = []

		self.dealer = Dealer(self.np_random)
		self.deck_str = cards2str(self.dealer.deck)

	def initiate(self, players):
		"""
		调用给庄家发牌和出牌
		"""
		landlord_id = self.dealer.calc_role(players)
		seed_cards = self.dealer.deck[-3:] # 地主牌
		seed_cards.sort(key=functools.cmp_to_key(ddz_sort_card_obj))
		self.seed_cards = cards2str(seed_cards)
		self.landlord_id = landlord_id
		self.current_player = landlord_id
		self.public = {
			'deck': self.deck_str,
			'seed_cards': self.seed_cards,
			'landlord': self.landlord_id,
			'trace': self.trace,
			'played_cards': ['' for _ in range(len(players))]
		}

	@staticmethod
	def cards_nd_array_to_str(nd_array_cards):
		"""
		将nd_array转换为str
		"""
		result = []
		for cards in nd_array_cards:
			_result = []
			for i, _ in enumerate(cards):
				if cards[i] != 0:
					_result.extend([CARD_RANK_STR[i] * cards[i]])
			result.append(''.join(_result))
		return result

	def update_public_info(self, action):
		"""
		更新信息
		:param action(str): 合法动作
		"""
		self.trace.append((self.current_player, action))
		if action != 'pass':
			for c in action:
				self.played_cards[self.current_player][CARD_RANK_STR_INDEX[c]] += 1
				if self.current_player == 0 and c in self.seed_cards:
					self.seed_cards = self.seed_cards.replace(c, '')
					self.public['seed_cards'] = self.seed_cards
			self.public['played_cards'] = self.cards_nd_array_to_str(self.played_cards)

	def proceed_round(self, player, action):
		"""
		调用另一个函数以保持一轮的运行
		:param player(obj): 斗地主循环器
		:param action(str): 一系列合法具体动作
		:return: 当前最大牌的玩家
		"""
		# 玩家有两个操作(出牌和捡牌)
		self.update_public_info(action)
		self.greater_player = player.play(action, self.greater_player)
		return self.greater_player

	def step_back(self, players):
		"""
		反转上一个动作
		:param players(list): 玩家对象列表
		:return: 最后的玩家ID和玩过的牌
		"""
		player_id, cards = self.trace.pop()
		self.current_player = player_id

		if (cards != 'pass'):
			for c in cards:
				self.played_cards[player_id][CARD_RANK_STR_INDEX[c]] -= 1
			self.public['played_cards'] = self.cards_nd_array_to_str(self.played_cards)

		greater_player_id = self.find_last_greater_player_id_in_trace()

		if greater_player_id is not None:
			self.greater_player = players[greater_player_id]
		else:
			self.greater_player = None

		return player_id, cards

	def find_last_greater_player_id_in_trace(self):
		"""
		在跟踪中找到最后一个greater_player的id
		:return:  trace中最后一个greater_player的id
		"""
		for i in range(len(self.trace) - 1, -1, -1):
			_id, action = self.trace[i]
			if action != 'pass':
				return _id
		return None

	def find_last_played_cards_in_trace(self, player_id):
		"""
		在trace中找到player_id最近玩过的牌
		:param player_id: 玩家id
		:return: 跟踪player_id最后玩过的牌
		"""
		for i in range(len(self.trace) - 1, -1, -1):
			_id, action = self.trace[i]
			if _id == player_id and action != 'pass':
				return action
		return None