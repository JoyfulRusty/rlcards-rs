# -*- coding: utf-8 -*-

import numpy as np

from reinforce.games.monster_v2.const import Card2Column
from reinforce.games.monster_v2.card import MonsterCards as Cards

PICK_CARD = Cards.PICK_CARD.val
UNIVERSAL_CARD = Cards.UNIVERSAL_CARD.val


class OneHot:
	"""
	One-hot encoding for a given set of values.
	"""

	@classmethod
	def cards2onehot(cls, hand_cards, is_bust=False):
		"""
		Hand Cards (played cards, picked cards remain cards and so on) Encode a value into a one-hot vector.
		"""
		if not hand_cards or is_bust:
			return np.zeros(shape=32, dtype=np.float32)
		matrix = np.zeros(shape=(4, 7), dtype=np.float32)
		universal_matrix = np.zeros(shape=4, dtype=np.float32)  # Magic and Pick
		hand_cards = [(card // 100, card % 100) for card in hand_cards]
		universal_count = 0
		for suit, card in hand_cards:
			if card == UNIVERSAL_CARD:
				universal_count += 1
				continue
			matrix[suit - 1][Card2Column[card]] = 1
		universal_matrix[:universal_count] = 1
		return np.concatenate((matrix.flatten("F"), universal_matrix))

	@classmethod
	def action2onehot(cls, legal_cards):
		"""
		Action Encode a value into a one-hot vector.
		"""
		if not legal_cards:
			return np.zeros(shape=33, dtype=np.float32)
		matrix = np.zeros(shape=(4, 7), dtype=np.float32)
		universal_matrix = np.zeros(shape=4, dtype=np.float32)
		pick_matrix = np.zeros(shape=1, dtype=np.float32)
		legal_cards = [(card // 100, card % 100) for card in legal_cards]
		universal_count = 0
		for suit, card in legal_cards:
			if card == PICK_CARD:
				pick_matrix[-1] = 1
			elif card == UNIVERSAL_CARD:
				universal_count += 1
			else:
				matrix[suit - 1][Card2Column[card]] = 1
		universal_matrix[:universal_count] = 1
		mp_matrix = np.concatenate((universal_matrix.flatten("F"), pick_matrix))
		return np.concatenate((matrix.flatten("F"), mp_matrix))

	@classmethod
	def action2tensor(cls, action):
		"""
		Action Encode a value into a one-hot vector.
		"""
		if not action:
			return np.zeros(shape=33, dtype=np.float32)
		matrix = np.zeros(shape=(4, 7), dtype=np.float32)
		pick_matrix = np.zeros(shape=1, dtype=np.float32)
		universal_matrix = np.zeros(shape=4, dtype=np.float32)
		suit, val = action // 100, action % 100
		if val == PICK_CARD:
			pick_matrix[-1] = 1
		elif val == UNIVERSAL_CARD:
			universal_matrix[0] = 1
		else:
			matrix[suit - 1][Card2Column[val]] = 1
		mp_matrix = np.concatenate((universal_matrix.flatten("F"), pick_matrix))
		return np.concatenate((matrix.flatten("F"), mp_matrix))

	@classmethod
	def nums2onehot(cls, num_left_cards, max_num_cards=32, is_bust=False):
		"""
		Cards nums Encode a value into a one-hot vector
		"""
		if not num_left_cards or is_bust:
			return np.zeros(shape=max_num_cards, dtype=np.float32)
		matrix = np.zeros(shape=max_num_cards, dtype=np.float32)
		matrix[num_left_cards - 1] = 1
		return matrix

	@classmethod
	def process_action_seq(cls, seq_list, length=12):
		"""
		Action Sequence Encode a value into a one-hot vector.
		History 4 moves a round
		"""
		seq_list = seq_list[-length:][:]
		if len(seq_list) < length:
			empty_seq = [[] for _ in range(length - len(seq_list))]
			empty_seq.extend(seq_list)
			seq_list = empty_seq
		return seq_list

	@classmethod
	def action_seq_list2array(cls, seq_list):
		"""
		Action Sequence List to Array Encode a value into a one-hot vector.
		"""
		seq_array = np.zeros(shape=(len(seq_list), 33), dtype=np.float32)
		for idx, seq in enumerate(seq_list):
			seq_array[idx, :] = cls.action2onehot(seq)
		return seq_array.reshape(len(seq_list), 33)