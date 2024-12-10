# -*- coding: utf-8 -*-


class BaseState:
	"""
	Base class for state model layers parameters representation.
	"""

	def __init__(self):
		"""
		Initialize the state model.
		"""
		# Current Player and Other Information
		self.__last_action = None
		self.__legal_actions = None

		# Initial Card Information
		self.__remain_cards = []
		self.__action_history = []
		self.__other_hand_cards = []
		self.__other_played_cards = []
		self.__other_picked_cards = []

	@property
	def last_action(self):
		"""
		Get the last action.
		"""
		return self.__last_action

	@last_action.setter
	def last_action(self, la):
		"""
		Set the last action.
		"""
		self.__last_action = la

	@property
	def legal_actions(self):
		"""
		Get the legal actions.
		"""
		return self.__legal_actions

	@legal_actions.setter
	def legal_actions(self, las):
		"""
		Set the legal actions.
		"""
		self.__legal_actions = las

	@property
	def remain_cards(self):
		"""
		Get the remaining cards.
		"""
		return self.__remain_cards

	@remain_cards.setter
	def remain_cards(self, rc):
		"""
		Set the remaining cards.
		"""
		self.__remain_cards = rc

	@property
	def action_history(self):
		"""
		Get the action history.
		"""
		return self.__action_history

	@action_history.setter
	def action_history(self, ahs):
		"""
		Set the action history.
		"""
		self.__action_history = ahs

	@property
	def other_hand_cards(self):
		"""
		Get the other player's hand cards.
		"""
		return self.__other_hand_cards

	@other_hand_cards.setter
	def other_hand_cards(self, ohc):
		"""
		Set the other player's hand cards.
		"""
		self.__other_hand_cards = ohc

	@property
	def other_played_cards(self):
		"""
		Get the other player's played cards.
		"""
		return self.__other_played_cards

	@other_played_cards.setter
	def other_played_cards(self, opc):
		"""
		Set the other player's played cards.
		"""
		self.__other_played_cards = opc

	@property
	def other_picked_cards(self):
		"""
		Get the other player's picked cards.
		"""
		return self.__other_picked_cards

	@other_picked_cards.setter
	def other_picked_cards(self, opc):
		"""
		Set the other player's picked cards.
		"""
		self.__other_picked_cards = opc

	def update_attrs(
			self,
			last_action=None,
			legal_actions=None,
			remain_cards=None,
			action_history=None,
			other_hand_cards=None,
			other_played_cards=None,
			other_picked_cards=None
		):
		"""
		Initialize Public the state model attributes.
		"""
		self.last_action = last_action
		self.legal_actions = legal_actions or []
		self.action_history = action_history or []
		self.remain_cards = remain_cards or []
		self.other_hand_cards = other_hand_cards or []
		self.other_played_cards = other_played_cards or []
		self.other_picked_cards = other_picked_cards or []

	def get_obs(self):
		"""
		Return the state model observation.
		"""
		raise NotImplementedError