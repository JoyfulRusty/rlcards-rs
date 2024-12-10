# -*- coding: utf-8 -*-

from typing import List

from reinforce.share.base_card import BaseCard


class BasePlayer:
	"""
	Base class for players
	"""

	def __init__(self, seat_id: int) -> None:
		"""
		Initialize a new BasePlayer
		:param seat_id: The seat number of the player
		"""
		self.__golds = 0
		self.__hand_cards = []
		self.__seat_id = seat_id

	@property
	def hand_cards(self) -> List[BaseCard]:
		"""
		Hand cards property getter
		"""
		return self.__hand_cards

	@hand_cards.setter
	def hand_cards(self, cards: List[BaseCard]) -> None:
		"""
		Hand cards property setter
		"""
		self.__hand_cards = cards

	@property
	def seat_id(self) -> int:
		"""
		Seat id property getter
		"""
		return self.__seat_id

	@seat_id.setter
	def seat_id(self, seat_id: int) -> None:
		"""
		Seat id property setter
		"""
		self.__seat_id = seat_id

	@property
	def golds(self) -> int:
		"""
		Golds property getter
		"""
		return self.__golds

	@golds.setter
	def golds(self, golds: int) -> None:
		"""
		Golds property setter
		"""
		self.__golds = golds