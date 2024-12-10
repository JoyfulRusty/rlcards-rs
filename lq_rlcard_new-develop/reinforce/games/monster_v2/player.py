# -*- coding: utf-8 -*-

from typing import List

from reinforce.share.base_card import BaseCard
from reinforce.share.base_player import BasePlayer


class MonsterPlayer(BasePlayer):
	"""
	Monster Player
	"""

	def __init__(self, seat_id: int) -> None:
		"""
		Initialize Monster Player
		:param seat_id: The seat ID of the player
		"""
		super().__init__(seat_id)
		self.is_out = False
		self.is_all = False
		self.picked_cards = []
		self.played_cards = []
		self.pick_rewards = 0.0

	def play_cards(self, action: BaseCard) -> None:
		"""
		Played cards
		"""
		action_idx = self.hand_cards.index(action)
		card = self.hand_cards.pop(action_idx)
		self.played_cards.append(card)

	def pick_cards(self, round_cards: List[BaseCard]) -> None:
		"""
		Picked cards
		"""
		self.picked_cards.extend(round_cards)