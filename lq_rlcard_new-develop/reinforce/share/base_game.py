# -*- coding: utf-8 -*-

class BaseGame:
	"""
	Base class for all games.
	"""

	def __init__(self):
		"""
		Initialize the game.
		"""
		self.__curr_p = None
		self.__players = None
		self.__winner_id = None
		self.__landlord_id = None
		self.__curr_seat_id = None
		self.__last_seat_id = None
		# Other settings
		self.__state = None
		self.__last_action = None
		# Game Flows Class
		self.__pokers = None
		# Cache list game round infos
		self.__traces = []
		self.__round_cards = []
		self.__remain_cards = []
		self.__legal_actions = []
		self.__action_history = []

	@property
	def curr_p(self):
		"""
		Get the current player.
		"""
		return self.__curr_p

	@curr_p.setter
	def curr_p(self, other_p):
		"""
		Set the current player.
		"""
		self.__curr_p = other_p

	@property
	def players(self):
		"""
		Get the players.
		"""
		return self.__players

	@players.setter
	def players(self, others_p):
		"""
		Set the players.
		"""
		self.__players = others_p

	@property
	def winner_id(self):
		"""
		Get the winner id.
		"""
		return self.__winner_id

	@winner_id.setter
	def winner_id(self, wid):
		"""
		Set the winner id.
		"""
		self.__winner_id = wid

	@property
	def landlord_id(self):
		"""
		Get the landlord id.
		"""
		return self.__landlord_id

	@landlord_id.setter
	def landlord_id(self, lid):
		"""
		Set the landlord id.
		"""
		self.__landlord_id = lid

	@property
	def curr_seat_id(self):
		"""
		Get the current seat id.
		"""
		return self.__curr_seat_id

	@curr_seat_id.setter
	def curr_seat_id(self, cs_id):
		"""
		Set the current seat id.
		"""
		self.__curr_seat_id = cs_id

	@property
	def last_seat_id(self):
		"""
		Get the last seat id.
		"""
		return self.__last_seat_id

	@last_seat_id.setter
	def last_seat_id(self, ls_id):
		"""
		Set the last seat id.
		"""
		self.__last_seat_id = ls_id

	@property
	def state(self):
		"""
		Get the state.
		"""
		return self.__state

	@state.setter
	def state(self, s):
		"""
		Set the state.
		"""
		self.__state = s

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
	def pokers(self):
		"""
		Get the pokers.
		"""
		return self.__pokers

	@pokers.setter
	def pokers(self, ps):
		"""
		Set the pokers.
		"""
		self.__pokers = ps

	@property
	def traces(self):
		"""
		Get the traces.
		"""
		return self.__traces

	@traces.setter
	def traces(self, ts):
		"""
		Set the traces.
		"""
		self.__traces = ts

	@property
	def round_cards(self):
		"""
		Get the round cards.
		"""
		return self.__round_cards

	@round_cards.setter
	def round_cards(self, rcs):
		"""
		Set the round cards.
		"""
		self.__round_cards = rcs

	@property
	def remain_cards(self):
		"""
		Get the remain cards.
		"""
		return self.__remain_cards

	@remain_cards.setter
	def remain_cards(self, rcs):
		"""
		Set the remain cards.
		"""
		self.__remain_cards = rcs

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

	def clear(self):
		"""
		Clear the game.
		"""
		self.__curr_p = None
		self.__players = None
		self.__winner_id = None
		self.__landlord_id = None
		self.__curr_seat_id = None
		self.__last_seat_id = None
		# Other settings
		self.__state = None
		self.__last_action = None
		# Game Flows Class
		self.__pokers = None
		# Cache list game round infos
		self.__traces = []
		self.__round_cards = []
		self.__remain_cards = []
		self.__legal_actions = []
		self.__action_history = []

	def reset(self):
		"""
		Reset the game.
		"""
		raise NotImplementedError

	def init_game(self):
		"""
		Initialize the game.
		"""
		raise NotImplementedError

	def step(self, action):
		"""
		Step the game by action.
		"""
		raise NotImplementedError

	def process_round(self):
		"""
		Process the round.
		"""
		raise NotImplementedError

	def payoffs(self):
		"""
		Get the payoffs of players.
		"""
		raise NotImplementedError