# -*- coding: utf-8 -*-

import numpy as np

from collections import Counter
from predict.ddz.xxc.const import Card2Column, NumOnes2Array, B_KING, S_KING


def get_obs(info_set):
	"""
	Obtain the observation for the player.
	"""
	if info_set.player_position == 'landlord':
		return _get_obs_landlord(info_set)
	elif info_set.player_position == 'landlord_up':
		return _get_obs_landlord_up(info_set)
	elif info_set.player_position == 'landlord_down':
		return _get_obs_landlord_down(info_set)
	else:
		raise ValueError('')

def _get_obs_landlord(info_set):
	"""
	Obtain the landlord features. See Table 4 in
	https://arxiv.org/pdf/2106.06135.pdf
	"""
	num_legal_actions = len(info_set.legal_actions)
	my_hand_cards = _cards2array(info_set.player_hand_cards)
	my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	other_hand_cards = _cards2array(info_set.other_hand_cards)
	other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	last_action = _cards2array(info_set.last_move)
	last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)
	
	my_action_batch = np.zeros(my_hand_cards_batch.shape)
	for j, action in enumerate(info_set.legal_actions):
		my_action_batch[j, :] = _cards2array(action)
	
	landlord_up_num_cards_left = _get_one_hot_array(
		info_set.num_cards_left_dict['landlord_up'], 17)
	landlord_up_num_cards_left_batch = np.repeat(
		landlord_up_num_cards_left[np.newaxis, :],
		num_legal_actions, axis=0)
	
	landlord_down_num_cards_left = _get_one_hot_array(
		info_set.num_cards_left_dict['landlord_down'], 17)
	landlord_down_num_cards_left_batch = np.repeat(
		landlord_down_num_cards_left[np.newaxis, :],
		num_legal_actions, axis=0)
	
	landlord_up_played_cards = _cards2array(
		info_set.played_cards['landlord_up'])
	landlord_up_played_cards_batch = np.repeat(
		landlord_up_played_cards[np.newaxis, :],
		num_legal_actions, axis=0)
	
	landlord_down_played_cards = _cards2array(
		info_set.played_cards['landlord_down'])
	landlord_down_played_cards_batch = np.repeat(
		landlord_down_played_cards[np.newaxis, :],
		num_legal_actions, axis=0)
	
	bomb_num = _get_one_hot_bomb(
		info_set.bomb_num)
	bomb_num_batch = np.repeat(
		bomb_num[np.newaxis, :],
		num_legal_actions, axis=0)
	
	x_batch = np.hstack((
		my_hand_cards_batch,
		other_hand_cards_batch,
		last_action_batch,
		landlord_up_played_cards_batch,
		landlord_down_played_cards_batch,
		landlord_up_num_cards_left_batch,
		landlord_down_num_cards_left_batch,
		bomb_num_batch,
		my_action_batch
	))
	x_no_action = np.hstack((
		my_hand_cards,
		other_hand_cards,
		last_action,
		landlord_up_played_cards,
		landlord_down_played_cards,
		landlord_up_num_cards_left,
		landlord_down_num_cards_left,
		bomb_num
	))
	z = _action_seq_list2array(_process_action_seq(info_set.card_play_action_seq))
	z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
	obs = {
		'position': 'landlord',
		'x_batch': x_batch.astype(np.float32),
		'z_batch': z_batch.astype(np.float32),
		'legal_actions': info_set.legal_actions,
		'x_no_action': x_no_action.astype(np.int8),
		'z': z.astype(np.int8),
	}
	return obs


def _get_obs_landlord_up(info_set):
	"""
	Obtain the landlord_up features. See Table 5 in
	https://arxiv.org/pdf/2106.06135.pdf
	"""
	num_legal_actions = len(info_set.legal_actions)
	my_hand_cards = _cards2array(info_set.player_hand_cards)
	my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	other_hand_cards = _cards2array(info_set.other_hand_cards)
	other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	last_action = _cards2array(info_set.last_move)
	last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)
	
	my_action_batch = np.zeros(my_hand_cards_batch.shape)
	for j, action in enumerate(info_set.legal_actions):
		my_action_batch[j, :] = _cards2array(action)
	
	last_landlord_action = _cards2array(info_set.last_move_dict['landlord'])
	last_landlord_action_batch = np.repeat(last_landlord_action[np.newaxis, :], num_legal_actions, axis=0)
	landlord_num_cards_left = _get_one_hot_array(info_set.num_cards_left_dict['landlord'], 20)
	landlord_num_cards_left_batch = np.repeat(landlord_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)
	
	landlord_played_cards = _cards2array(info_set.played_cards['landlord'])
	landlord_played_cards_batch = np.repeat(landlord_played_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	last_teammate_action = _cards2array(info_set.last_move_dict['landlord_down'])
	last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)
	teammate_num_cards_left = _get_one_hot_array(info_set.num_cards_left_dict['landlord_down'], 17)
	teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)
	
	teammate_played_cards = _cards2array(info_set.played_cards['landlord_down'])
	teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	bomb_num = _get_one_hot_bomb(info_set.bomb_num)
	bomb_num_batch = np.repeat(bomb_num[np.newaxis, :], num_legal_actions, axis=0)
	
	x_batch = np.hstack((
		my_hand_cards_batch,
		other_hand_cards_batch,
		landlord_played_cards_batch,
		teammate_played_cards_batch,
		last_action_batch,
		last_landlord_action_batch,
		last_teammate_action_batch,
		landlord_num_cards_left_batch,
		teammate_num_cards_left_batch,
		bomb_num_batch,
		my_action_batch
	))
	x_no_action = np.hstack((
		my_hand_cards,
		other_hand_cards,
		landlord_played_cards,
		teammate_played_cards,
		last_action,
		last_landlord_action,
		last_teammate_action,
		landlord_num_cards_left,
		teammate_num_cards_left,
		bomb_num
	))
	z = _action_seq_list2array(_process_action_seq(info_set.card_play_action_seq))
	z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
	obs = {
		'position': 'landlord_up',
		'x_batch': x_batch.astype(np.float32),
		'z_batch': z_batch.astype(np.float32),
		'legal_actions': info_set.legal_actions,
		'x_no_action': x_no_action.astype(np.int8),
		'z': z.astype(np.int8),
	}
	return obs


def _get_obs_landlord_down(info_set):
	"""
	Obtain the landlord_down features. See Table 5 in
	https://arxiv.org/pdf/2106.06135.pdf
	"""
	num_legal_actions = len(info_set.legal_actions)
	my_hand_cards = _cards2array(info_set.player_hand_cards)
	my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	other_hand_cards = _cards2array(info_set.other_hand_cards)
	other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)
	
	last_action = _cards2array(info_set.last_move)
	last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)
	
	my_action_batch = np.zeros(my_hand_cards_batch.shape)
	for j, action in enumerate(info_set.legal_actions):
		my_action_batch[j, :] = _cards2array(action)
	
	last_landlord_action = _cards2array(
		info_set.last_move_dict['landlord'])
	last_landlord_action_batch = np.repeat(
		last_landlord_action[np.newaxis, :],
		num_legal_actions, axis=0)
	landlord_num_cards_left = _get_one_hot_array(
		info_set.num_cards_left_dict['landlord'], 20)
	landlord_num_cards_left_batch = np.repeat(
		landlord_num_cards_left[np.newaxis, :],
		num_legal_actions, axis=0)
	
	landlord_played_cards = _cards2array(
		info_set.played_cards['landlord'])
	landlord_played_cards_batch = np.repeat(
		landlord_played_cards[np.newaxis, :],
		num_legal_actions, axis=0)
	
	last_teammate_action = _cards2array(
		info_set.last_move_dict['landlord_up'])
	last_teammate_action_batch = np.repeat(
		last_teammate_action[np.newaxis, :],
		num_legal_actions, axis=0)
	teammate_num_cards_left = _get_one_hot_array(
		info_set.num_cards_left_dict['landlord_up'], 17)
	teammate_num_cards_left_batch = np.repeat(
		teammate_num_cards_left[np.newaxis, :],
		num_legal_actions, axis=0)
	
	teammate_played_cards = _cards2array(
		info_set.played_cards['landlord_up'])
	teammate_played_cards_batch = np.repeat(
		teammate_played_cards[np.newaxis, :],
		num_legal_actions, axis=0)
	
	bomb_num = _get_one_hot_bomb(
		info_set.bomb_num)
	bomb_num_batch = np.repeat(
		bomb_num[np.newaxis, :],
		num_legal_actions, axis=0)
	
	x_batch = np.hstack((
		my_hand_cards_batch,
		other_hand_cards_batch,
		landlord_played_cards_batch,
		teammate_played_cards_batch,
		last_action_batch,
		last_landlord_action_batch,
		last_teammate_action_batch,
		landlord_num_cards_left_batch,
		teammate_num_cards_left_batch,
		bomb_num_batch,
		my_action_batch
	))
	x_no_action = np.hstack((
		my_hand_cards,
		other_hand_cards,
		landlord_played_cards,
		teammate_played_cards,
		last_action,
		last_landlord_action,
		last_teammate_action,
		landlord_num_cards_left,
		teammate_num_cards_left,
		bomb_num
	))
	z = _action_seq_list2array(_process_action_seq(
		info_set.card_play_action_seq))
	z_batch = np.repeat(
		z[np.newaxis, :, :],
		num_legal_actions, axis=0)
	obs = {
		'position': 'landlord_down',
		'x_batch': x_batch.astype(np.float32),
		'z_batch': z_batch.astype(np.float32),
		'legal_actions': info_set.legal_actions,
		'x_no_action': x_no_action.astype(np.int8),
		'z': z.astype(np.int8),
	}
	return obs

def _get_one_hot_array(num_left_cards, max_num_cards):
	"""
	A utility function to obtain one-hot encoding
	"""
	one_hot = np.zeros(max_num_cards)
	one_hot[num_left_cards - 1] = 1
	
	return one_hot

def _cards2array(list_cards):
	"""
	A utility function that transforms the actions, i.e.,
	A list of integers into card matrix. Here we remove
	the six entries that are always zero and flatten the
	representations.
	"""
	if not list_cards:
		return np.zeros(54, dtype=np.int8)
	matrix = np.zeros([4, 13], dtype=np.int8)
	jokers = np.zeros(2, dtype=np.int8)
	counter = Counter(list_cards)
	for card, num_times in counter.items():
		if int(card) < 17:
			matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
		elif card == S_KING:
			jokers[0] = 1
		elif card == B_KING:
			jokers[1] = 1
	return np.concatenate((matrix.flatten('F'), jokers))


def _action_seq_list2array(action_seq_list):
	"""
	A utility function to encode the historical moves.
	We encode the historical 15 actions. If there is
	no 15 actions, we pad the features with 0. Since
	three moves is a round in Dou Di zhu, we concatenate
	the representations for each consecutive three moves.
	Finally, we obtain a 5x162 matrix, which will be fed
	into LSTM for encoding.
	"""
	action_seq_array = np.zeros((len(action_seq_list), 54))
	for row, list_cards in enumerate(action_seq_list):
		action_seq_array[row, :] = _cards2array(list_cards)
	action_seq_array = action_seq_array.reshape(5, 162)
	return action_seq_array


def _process_action_seq(sequence, length=15):
	"""
	A utility function encoding historical moves. We
	encode 15 moves. If there is no 15 moves, we pad
	with zeros.
	"""
	sequence = sequence[-length:].copy()
	if len(sequence) < length:
		empty_sequence = [[] for _ in range(length - len(sequence))]
		empty_sequence.extend(sequence)
		sequence = empty_sequence
	return sequence


def _get_one_hot_bomb(bomb_num):
	"""
	A utility function to encode the number of bombs
	into one-hot representation.
	"""
	one_hot = np.zeros(15)
	one_hot[bomb_num] = 1
	return one_hot