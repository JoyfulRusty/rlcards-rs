# -*- coding: utf-8 -*-

import random

from typing import List, Union

from reinforce.share.base_card import BaseCard


class BasePoker:
	"""
	Base class for all poker games.
	"""

	CARDS_ENUM: BaseCard

	def __init__(self) -> None:
		"""
		Initialize a new poker game.
		"""
		self.__cursor = 0
		# Don't Contain Extra Cards
		self.__cards = self.CARDS_ENUM.all_cards()
		self.__cards_count = len(self.__cards)

	@property
	def cursor(self) -> int:
		"""
		Get the current cursor position.
		"""
		return self.__cursor

	@cursor.setter
	def cursor(self, val: int) -> None:
		"""
		Set the current cursor position.
		"""
		self.__cursor = val

	@property
	def cards(self) -> List[BaseCard]:
		"""
		Get the current cards.
		"""
		return self.__cards

	@cards.setter
	def cards(self, vals: List[BaseCard]) -> None:
		"""
		Set the current cards.
		"""
		self.__cards = vals

	@property
	def all_cards(self) -> List[BaseCard]:
		"""
		Get all cards and Contains Extra Cards.
		"""
		return self.__cards + self.CARDS_ENUM.extra_cards()

	@property
	def cards_count(self) -> int:
		"""
		Get the current cards count.
		"""
		return self.__cards_count

	@classmethod
	def find_card_by_val(cls, card_val: int) -> BaseCard:
		"""
		Find a card by its value.
		"""
		card = cls.find_card_by_val(card_val)
		if not card:
			raise ValueError(f"Card with value {card_val} not found.")
		return card

	def swap_cards(self, start_idx: int, end_idx: int) -> None:
		"""
		Swap cards between two positions.
		"""
		self.__cards[start_idx: end_idx + 1] = self.__cards[end_idx:start_idx - 1: -1]

	def shuffle_cards(self) -> None:
		"""
		Shuffle the cards.
		"""
		self.__cursor = 0
		random.shuffle(self.__cards)
		half_idx = self.__cards_count // 2
		self.swap_cards(half_idx, self.__cards_count)
		random.shuffle(self.__cards)

	def pop(self) -> Union[int, BaseCard]:
		"""
		Pop a card from the top of the deck.
		"""
		if self.__cursor >= self.__cards_count:
			return 0
		card = self.__cards[self.__cursor]
		self.__cursor += 1
		return card

	def deal_cards(self, player_nums: int = 0, card_nums: int = 0) -> List[List[BaseCard]]:
		"""
		Deal cards to players.
		:param player_nums: The number of players.
		:param card_nums: The number of cards to deal to each player.
		"""
		extra_list = []
		all_cards = [[] for _ in range(player_nums)]
		# Deal cards to each player
		self.shuffle_cards()
		cards = [self.pop() for _ in range(player_nums * card_nums)]
		all_cards = [all_cards[i] + cards[i * card_nums:(i + 1) * card_nums] for i in range(player_nums)]
		return all_cards

	@property
	def left_count(self) -> int:
		"""
		Get the number of cards left in the deck.
		"""
		return max(self.__cards_count - self.__cursor, 0)

	@property
	def remain_cards(self) -> List[BaseCard]:
		"""
		Get the remaining cards in the deck.
		"""
		return self.__cards[self.__cursor:]