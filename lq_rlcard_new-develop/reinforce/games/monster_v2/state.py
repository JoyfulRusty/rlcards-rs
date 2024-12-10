# -*- coding: utf-8 -*-

import numpy as np

from typing import Dict, Callable
from reinforce.share.base_state import BaseState
from reinforce.games.monster_v2.onehot import OneHot


# todo: build model layers states parameters

class MonsterState(BaseState):
	"""
	State of the game.
	"""

	def __init__(self):
		"""
		Initializes the state.
		"""
		super().__init__()
		self.__seat_id = 0
		self.__hand_cards = []
		self.__played_cards = []
		self.__bust_infos = {}
		self.__round_cards = []
		self.__other_left_cards = {}

	@property
	def seat_id(self):
		"""
		Returns the seat id.
		"""
		return self.__seat_id

	@seat_id.setter
	def seat_id(self, sd):
		"""
		Sets the seat id.
		"""
		self.__seat_id = sd

	@property
	def hand_cards(self):
		"""
		Returns the hand cards.
		"""
		return self.__hand_cards

	@hand_cards.setter
	def hand_cards(self, hc):
		"""
		Sets the hand cards.
		"""
		self.__hand_cards = hc

	@property
	def played_cards(self):
		"""
		Returns the played cards.
		"""
		return self.__played_cards

	@played_cards.setter
	def played_cards(self, pc):
		"""
		Sets the played cards.
		"""
		self.__played_cards = pc

	@property
	def bust_infos(self):
		"""
		Returns the bust infos.
		"""
		return self.__bust_infos

	@bust_infos.setter
	def bust_infos(self, bs):
		"""
		Sets the bust infos.
		"""
		self.__bust_infos = bs

	@property
	def round_cards(self):
		"""
		Returns the round cards.
		"""
		return self.__round_cards

	@round_cards.setter
	def round_cards(self, rc):
		"""
		Sets the round cards.
		"""
		self.__round_cards = rc

	@property
	def other_left_cards(self):
		"""
		Returns the other left cards.
		"""
		return self.__other_left_cards

	@other_left_cards.setter
	def other_left_cards(self, olc):
		"""
		Sets the other left cards.
		"""
		self.__other_left_cards = olc

	def init_attrs(
			self,
			seat_id,
			hand_cards,
			played_cards,
			last_action,
			legal_actions,
			round_cards,
			remain_cards,
			action_history,
			other_played_cards,
			other_left_cards,
			bust_infos,
		):
		"""
		Initializes Private the attributes.
		"""
		# todo: init attrs
		self.update_attrs(
			last_action=last_action,
			legal_actions=legal_actions,
			remain_cards=remain_cards,
			other_played_cards=other_played_cards,
		)

		# todo: add attributes
		self.seat_id = seat_id
		self.hand_cards = hand_cards
		self.played_cards = played_cards
		self.bust_infos = bust_infos
		self.round_cards = round_cards
		self.other_left_cards = other_left_cards

	def get_obs(self):
		"""
		Returns the observation.
		"""
		obs_func: Dict[int, Callable] = {
			0: self.get_obs_by_down,
			1: self.get_obs_by_right,
			2: self.get_obs_by_up,
			3: self.get_obs_by_left,
		}
		return obs_func[self.seat_id]()

	def get_obs_by_down(self):
		"""
		Returns the observation for the down player.
		"""
		# 1.legal_action
		nums_legal_actions = len(self.legal_actions)

		# 2.hand_cards and played_cards batch
		hand_cards = OneHot.cards2onehot(self.hand_cards, self.bust_infos["down"])
		hand_cards_batch = np.repeat(hand_cards[np.newaxis, :], nums_legal_actions, axis=0)

		played_cards = OneHot.cards2onehot(self.played_cards)
		played_cards_batch = np.repeat(played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 3.other_left_cards and other_played_cards batch
		right_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["right"],
			is_bust=self.bust_infos["right"]
		)
		right_hand_nums_batch = np.repeat(right_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		right_played_cards = OneHot.cards2onehot(self.other_played_cards["right"])
		right_played_cards_batch = np.repeat(right_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		up_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["up"],
			is_bust=self.bust_infos["up"]
		)
		up_hand_nums_batch = np.repeat(up_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		up_played_cards = OneHot.cards2onehot(self.other_played_cards["up"])
		up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		left_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["left"],
			is_bust=self.bust_infos["left"])
		left_hand_nums_batch = np.repeat(left_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		left_played_cards = OneHot.cards2onehot(self.other_played_cards["left"])
		left_played_cards_batch = np.repeat(left_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 4.remain_cards and batch
		remain_cards = OneHot.cards2onehot(self.remain_cards)
		remain_cards_batch = np.repeat(remain_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 5.last_cards and batch
		last_action = OneHot.action2onehot(self.last_action)
		last_action_batch = np.repeat(last_action[np.newaxis, :], nums_legal_actions, axis=0)

		legal_actions_batch = np.zeros(last_action_batch.shape)
		for idx, action in enumerate(self.legal_actions):
			legal_actions_batch[idx, :] = OneHot.action2onehot([action])

		# 6.round_cards and batch
		round_cards = OneHot.cards2onehot(self.round_cards)
		round_cards_batch = np.repeat(round_cards[np.newaxis, :], nums_legal_actions, axis=0)
		round_nums = OneHot.nums2onehot(len(self.round_cards))
		round_nums_batch = np.repeat(round_nums[np.newaxis, :], nums_legal_actions, axis=0)

		x_no_action = np.hstack((
			hand_cards,
			played_cards,
			right_hand_nums,
			right_played_cards,
			up_hand_nums,
			up_played_cards,
			left_hand_nums,
			left_played_cards,
			last_action,
			remain_cards,
			round_cards,
			round_nums,
		))

		x_batch = np.hstack((
			hand_cards_batch,
			played_cards_batch,
			right_hand_nums_batch,
			right_played_cards_batch,
			up_hand_nums_batch,
			up_played_cards_batch,
			left_hand_nums_batch,
			left_played_cards_batch,
			last_action_batch,
			legal_actions_batch,
			remain_cards_batch,
			round_cards_batch,
			round_nums_batch,
		))

		# 8.action_history and batch
		z = OneHot.action_seq_list2array(OneHot.process_action_seq(self.action_history))
		z_batch = np.repeat(z[np.newaxis, :, :], nums_legal_actions, axis=0)

		obs = {
			"position": "down",
			"x_batch": x_batch.astype(np.float32),
			"z_batch": z_batch.astype(np.float32),
			"legal_actions": self.legal_actions,
			"last_action": self.last_action,
			"x_no_action": x_no_action.astype(np.int8),
			"z": z.astype(np.int8),
		}

		# clear
		self.clear()

		return obs

	def get_obs_by_right(self):
		"""
		Returns the observation for the right player.
		"""
		# 1.legal_action
		nums_legal_actions = len(self.legal_actions)

		# 2.hand_cards and played_cards batch
		hand_cards = OneHot.cards2onehot(self.hand_cards, self.bust_infos["right"])
		hand_cards_batch = np.repeat(hand_cards[np.newaxis, :], nums_legal_actions, axis=0)

		played_cards = OneHot.cards2onehot(self.played_cards)
		played_cards_batch = np.repeat(played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 3.other_left_cards and other_played_cards batch
		up_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["up"],
			is_bust=self.bust_infos["up"]
		)
		up_hand_nums_batch = np.repeat(up_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		up_played_cards = OneHot.cards2onehot(self.other_played_cards["up"])
		up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		left_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["left"],
			is_bust=self.bust_infos["left"]
		)
		left_hand_nums_batch = np.repeat(left_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		left_played_cards = OneHot.cards2onehot(self.other_played_cards["left"])
		left_played_cards_batch = np.repeat(left_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		down_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["down"],
			is_bust=self.bust_infos["down"]
		)
		down_hand_nums_batch = np.repeat(down_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		down_played_cards = OneHot.cards2onehot(self.other_played_cards["down"])
		down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 4.remain_cards and batch
		remain_cards = OneHot.cards2onehot(self.remain_cards)
		remain_cards_batch = np.repeat(remain_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 5.last_cards and batch
		last_action = OneHot.action2onehot(self.last_action)
		last_action_batch = np.repeat(last_action[np.newaxis, :], nums_legal_actions, axis=0)

		legal_actions_batch = np.zeros(last_action_batch.shape)
		for idx, action in enumerate(self.legal_actions):
			legal_actions_batch[idx, :] = OneHot.action2onehot([action])

		# 6.round_cards and batch
		round_cards = OneHot.cards2onehot(self.round_cards)
		round_cards_batch = np.repeat(round_cards[np.newaxis, :], nums_legal_actions, axis=0)
		round_nums = OneHot.nums2onehot(len(self.round_cards))
		round_nums_batch = np.repeat(round_nums[np.newaxis, :], nums_legal_actions, axis=0)

		x_no_action = np.hstack((
			hand_cards,
			played_cards,
			up_hand_nums,
			up_played_cards,
			left_hand_nums,
			left_played_cards,
			down_hand_nums,
			down_played_cards,
			last_action,
			remain_cards,
			round_cards,
			round_nums,
		))

		x_batch = np.hstack((
			hand_cards_batch,
			played_cards_batch,
			up_hand_nums_batch,
			up_played_cards_batch,
			left_hand_nums_batch,
			left_played_cards_batch,
			down_hand_nums_batch,
			down_played_cards_batch,
			last_action_batch,
			legal_actions_batch,
			remain_cards_batch,
			round_cards_batch,
			round_nums_batch
		))

		# 8.action_history and batch
		z = OneHot.action_seq_list2array(OneHot.process_action_seq(self.action_history))
		z_batch = np.repeat(z[np.newaxis, :, :], nums_legal_actions, axis=0)

		obs = {
			"position": "right",
			"x_batch": x_batch.astype(np.float32),
			"z_batch": z_batch.astype(np.float32),
			"legal_actions": self.legal_actions,
			"last_action": self.last_action,
			"x_no_action": x_no_action.astype(np.int8),
			"z": z.astype(np.int8),
		}

		# clear
		self.clear()

		return obs

	def get_obs_by_up(self):
		"""
		Returns the observation for the up player.
		"""

		# 1.legal_action
		nums_legal_actions = len(self.legal_actions)

		# 2.hand_cards and played_cards batch
		hand_cards = OneHot.cards2onehot(self.hand_cards, self.bust_infos["up"])
		hand_cards_batch = np.repeat(hand_cards[np.newaxis, :], nums_legal_actions, axis=0)

		played_cards = OneHot.cards2onehot(self.played_cards)
		played_cards_batch = np.repeat(played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 3.other_left_cards and other_played_cards batch
		left_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["left"],
			is_bust=self.bust_infos["left"]
		)
		left_hand_nums_batch = np.repeat(left_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		left_played_cards = OneHot.cards2onehot(self.other_played_cards["left"])
		left_played_cards_batch = np.repeat(left_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		down_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["down"],
			is_bust=self.bust_infos["down"]
		)
		down_hand_nums_batch = np.repeat(down_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		down_played_cards = OneHot.cards2onehot(self.other_played_cards["down"])
		down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		right_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["right"],
			is_bust=self.bust_infos["right"]
		)
		right_hand_nums_batch = np.repeat(right_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		right_played_cards = OneHot.cards2onehot(self.other_played_cards["right"])
		right_played_cards_batch = np.repeat(right_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 4.remain_cards and batch
		remain_cards = OneHot.cards2onehot(self.remain_cards)
		remain_cards_batch = np.repeat(remain_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 5.last_cards and batch
		last_action = OneHot.action2onehot(self.last_action)
		last_action_batch = np.repeat(last_action[np.newaxis, :], nums_legal_actions, axis=0)

		legal_actions_batch = np.zeros(last_action_batch.shape)
		for idx, action in enumerate(self.legal_actions):
			legal_actions_batch[idx, :] = OneHot.action2onehot([action])

		# 6.round_cards and batch
		round_cards = OneHot.cards2onehot(self.round_cards)
		round_cards_batch = np.repeat(round_cards[np.newaxis, :], nums_legal_actions, axis=0)
		round_nums = OneHot.nums2onehot(len(self.round_cards))
		round_nums_batch = np.repeat(round_nums[np.newaxis, :], nums_legal_actions, axis=0)

		x_no_action = np.hstack((
			hand_cards,
			played_cards,
			left_hand_nums,
			left_played_cards,
			down_hand_nums,
			down_played_cards,
			right_hand_nums,
			right_played_cards,
			last_action,
			remain_cards,
			round_cards,
			round_nums
		))

		x_batch = np.hstack((
			hand_cards_batch,
			played_cards_batch,
			left_hand_nums_batch,
			left_played_cards_batch,
			down_hand_nums_batch,
			down_played_cards_batch,
			right_hand_nums_batch,
			right_played_cards_batch,
			last_action_batch,
			legal_actions_batch,
			remain_cards_batch,
			round_cards_batch,
			round_nums_batch
		))

		# 8.action_history and batch
		z = OneHot.action_seq_list2array(OneHot.process_action_seq(self.action_history))
		z_batch = np.repeat(z[np.newaxis, :, :], nums_legal_actions, axis=0)

		obs = {
			"position": "up",
			"x_batch": x_batch.astype(np.float32),
			"z_batch": z_batch.astype(np.float32),
			"legal_actions": self.legal_actions,
			"last_action": self.last_action,
			"x_no_action": x_no_action.astype(np.int8),
			"z": z.astype(np.int8),
		}

		# clear
		self.clear()

		return obs

	def get_obs_by_left(self):
		"""
		Returns the observation for the left player.
		"""
		# 1.legal_action
		nums_legal_actions = len(self.legal_actions)

		# 2.hand_cards and played_cards batch
		hand_cards = OneHot.cards2onehot(self.hand_cards, self.bust_infos["left"])
		hand_cards_batch = np.repeat(hand_cards[np.newaxis, :], nums_legal_actions, axis=0)

		played_cards = OneHot.cards2onehot(self.played_cards)
		played_cards_batch = np.repeat(played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 3.other_left_cards and other_played_cards batch
		down_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["down"],
			is_bust=self.bust_infos["down"]
		)
		down_hand_nums_batch = np.repeat(down_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		down_played_cards = OneHot.cards2onehot(self.other_played_cards["down"])
		down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		right_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["right"],
			is_bust=self.bust_infos["right"]
		)
		right_hand_nums_batch = np.repeat(right_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		right_played_cards = OneHot.cards2onehot(self.other_played_cards["right"])
		right_played_cards_batch = np.repeat(right_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		up_hand_nums = OneHot.nums2onehot(
			num_left_cards=self.other_left_cards["up"],
			is_bust=self.bust_infos["up"]
		)
		up_hand_nums_batch = np.repeat(up_hand_nums[np.newaxis, :], nums_legal_actions, axis=0)
		up_played_cards = OneHot.cards2onehot(self.other_played_cards["up"])
		up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 4.remain_cards and batch
		remain_cards = OneHot.cards2onehot(self.remain_cards)
		remain_cards_batch = np.repeat(remain_cards[np.newaxis, :], nums_legal_actions, axis=0)

		# 5.last_cards and batch
		last_action = OneHot.action2onehot(self.last_action)
		last_action_batch = np.repeat(last_action[np.newaxis, :], nums_legal_actions, axis=0)

		legal_actions_batch = np.zeros(last_action_batch.shape)
		for idx, action in enumerate(self.legal_actions):
			legal_actions_batch[idx, :] = OneHot.action2onehot([action])

		# 6.round_cards and batch
		round_cards = OneHot.cards2onehot(self.round_cards)
		round_cards_batch = np.repeat(round_cards[np.newaxis, :], nums_legal_actions, axis=0)
		round_nums = OneHot.nums2onehot(len(self.round_cards))
		round_nums_batch = np.repeat(round_nums[np.newaxis, :], nums_legal_actions, axis=0)

		x_no_action = np.hstack((
			hand_cards,
			played_cards,
			down_hand_nums,
			down_played_cards,
			right_hand_nums,
			right_played_cards,
			up_hand_nums,
			up_played_cards,
			last_action,
			remain_cards,
			round_cards,
			round_nums
		))

		x_batch = np.hstack((
			hand_cards_batch,
			played_cards_batch,
			down_hand_nums_batch,
			down_played_cards_batch,
			right_hand_nums_batch,
			right_played_cards_batch,
			up_hand_nums_batch,
			up_played_cards_batch,
			last_action_batch,
			legal_actions_batch,
			remain_cards_batch,
			round_cards_batch,
			round_nums_batch
		))

		# 8.action_history and batch
		z = OneHot.action_seq_list2array(OneHot.process_action_seq(self.action_history))
		z_batch = np.repeat(z[np.newaxis, :, :], nums_legal_actions, axis=0)

		obs = {
			"position": "left",
			"x_batch": x_batch.astype(np.float32),
			"z_batch": z_batch.astype(np.float32),
			"legal_actions": self.legal_actions,
			"last_action": self.last_action,
			"x_no_action": x_no_action.astype(np.int8),
			"z": z.astype(np.int8),
		}

		# clear
		self.clear()

		return obs

	def clear(self):
		"""
		Clear the state
		"""
		# clear the base state
		self.last_action = None
		self.legal_actions = None
		self.remain_cards = []
		self.action_history = []
		self.other_played_cards = []

		# Clear current state
		self.seat_id = 0
		self.hand_cards = []
		self.played_cards = []
		self.bust_infos = {}
		self.round_cards = []