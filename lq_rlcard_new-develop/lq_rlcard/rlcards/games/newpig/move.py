# -*- coding: utf-8 -*-

from rlcards.const.pig.const import FIRST_PLAY, HAND_NUM

class MovesGenerator:
	"""
	生成动作
	"""
	def __init__(self, player):
		self.player = player
		self.position = player.player_id
		self.card_list = player.curr_hand_cards
		self.light_cards = player.light_cards or []  # 当前玩家亮的牌
		self.played_cards = player.played_cards or []  # 当前玩家打过的牌
		self.suit_cards = {}  # 以花色区分卡牌
		self.played_suit_cards = {}  # 打过的牌以花色区分

	def get_suit_cards(self, suit):
		"""
		获取以花色区分的牌
		"""
		if not self.suit_cards:
			for card in self.card_list:
				self.suit_cards.setdefault(card // 100, []).append(card)
		return self.suit_cards.get(suit) or []

	def get_suit_cards_by_played(self, suit):
		"""
		获取以花色区分打过的牌
		"""
		if not self.played_suit_cards:
			for card in self.played_cards:
				self.played_suit_cards.setdefault(card // 100, []).append(card)
		return self.played_suit_cards.get(suit) or []

	def limit_cp_to_light_cards(self, must_suit=None):
		"""
		亮牌后，若有亮牌花色的其他卡牌，第一次不得出亮的卡牌
		"""
		if not self.light_cards:
			return self.card_list
		if not set(self.card_list).intersection(self.light_cards):
			return self.card_list
		can_cards_list = self.card_list[:]
		suit_list = [card // 100 for card in self.light_cards]
		if must_suit:
			if must_suit not in suit_list:
				return can_cards_list
			suit_idx = suit_list.index(must_suit)
			suit_cards = self.get_suit_cards(must_suit)
			# 亮的卡牌花色大于1，且未打过该花色的卡牌
			if len(suit_cards) > 1 and not self.get_suit_cards_by_played(must_suit):
				can_cards_list.remove(self.light_cards[suit_idx])
		else:
			for i, suit in enumerate(suit_list):
				suit_cards = self.get_suit_cards(suit)
				# 亮的卡牌花色大于1，且未打过该花色的牌
				if len(suit_cards) > 1 and not self.get_suit_cards_by_played(suit):
					can_cards_list.remove(self.light_cards[i])

		return can_cards_list

	def gen_can_play_cards(self, must_suit=None) -> list:
		"""
		找出所有能出的合法卡牌，must_suit表示优先出的花色
		"""
		# 首出
		if len(self.card_list) == HAND_NUM:
			if FIRST_PLAY in self.card_list:
				return [FIRST_PLAY]
		# 第一位玩家出牌，判断玩家是否出过亮牌的同一种花色卡牌
		if not must_suit:
			can_cards_list = self.limit_cp_to_light_cards()
			return [card for card in can_cards_list]
		# 跟牌，若未出过亮过的牌，且该花色的牌数>1，则亮的卡牌不能打
		can_cards_list = self.limit_cp_to_light_cards()
		# 同花色卡牌
		can_actions = []
		for card in can_cards_list:
			if self.same_suit(card, must_suit):
				can_actions.append(card)
		return can_actions or [card for card in can_cards_list]  # 无同花色则任意出

	def gen_moves(self):
		"""
		从给定的牌中生成所有可能的动作
		"""
		moves = []
		moves.extend(self.gen_can_play_cards())  # 主动出
		return moves

	@staticmethod
	def same_suit(card, must_suit):
		return card // 100 == must_suit