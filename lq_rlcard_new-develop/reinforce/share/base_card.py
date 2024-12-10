# -*- coding: utf-8 -*-

from enum import IntEnum


class BaseCard(IntEnum):
	"""
	Base class for all cards.
	"""

	def __new__(cls, card, suit, val):
		"""
		Create a new card.
		:param card: The card (e.g. '2', 'A', 'K', 'Q', 'J', 'T').
		:param suit: The suit (e.g. 's' = Spades, 'h' = Hearts, 'd' = Diamonds, 'c' = Clubs).
		:param val: The value of the card (e.g. 2-14).
		"""
		obj = int.__new__(cls, card)
		obj.val = val  # 牌值
		obj.suit = suit  # 花色
		obj._value_ = card
		return obj

	@classmethod
	def find_member_by_val(cls, val: int):
		"""
		Find a card by its value.
		"""
		return cls._value2member_map_.get(val)

	@classmethod
	def all_cards(cls):
		"""
		Get all cards.
		"""
		raise NotImplementedError

	@classmethod
	def all_suits(cls):
		"""
		Get all suits.
		"""
		raise NotImplementedError

	@classmethod
	def extra_cards(cls):
		"""
		Get extra cards.
		"""
		raise NotImplementedError