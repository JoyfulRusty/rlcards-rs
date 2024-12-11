# -*- coding: utf-8 -*-

import numpy as np

from collections import Counter
from rlcards.const.gymahjong import const
from rlcards.games.gymahjong.card import GyMahjongCard as Card

def make_cards():
	"""
	制作108张卡牌+动作索引
	"""
	num = 0
	card_encoding_dict = {}
	for card in const.ALL_CARDS:
		card_encoding_dict[card] = num
		num += 1

	return add_actions(card_encoding_dict, num)

def add_actions(encode_dict, num):
	"""
	构建对应动作
	"""
	encode_dict[const.ActionType.GUO] = num
	encode_dict[const.ActionType.PONG] = num + 1
	encode_dict[const.ActionType.MING_GONG] = num + 2
	encode_dict[const.ActionType.SUO_GONG] = num + 3
	encode_dict[const.ActionType.AN_GONG] = num + 4
	encode_dict[const.ActionType.BAO_TING] = num + 5
	encode_dict[const.ActionType.KAI_HU] = num + 6

	return encode_dict

# 构建卡牌和动作索引映射
new_card_encoding_dict = make_cards()
card_decoding_dict = {new_card_encoding_dict[key]: key for key in new_card_encoding_dict.keys()}

def init_deck():
	"""
	实例化卡牌对象
	"""
	deck = []
	index_num = 0
	for card_value in const.ALL_CARDS:
		card_type = const.CARD_TYPES[card_value // 10]
		card = Card(card_type, card_value)
		card.set_index_num(index_num)
		index_num += 1
		deck.append(card)
	deck = deck * 4
	return deck

def compare2cards(card1, card2):
	"""
	比较两张卡牌值大小，并按从小到大排序
	"""
	key = []
	for card in [card1, card2]:
		key.append(const.CARD_VALUES.index(str(card)))
	if key[0] > key[1]:
		return 1
	if key[0] < key[1]:
		return -1
	return 0

# TODO: 编码卡牌
def encode_cards(cards):
	"""
	编码手牌
	"""
	matrix = np.zeros((27, 4), dtype=np.float32)
	for card in list(set(cards)):
		index = new_card_encoding_dict[card]
		nums = cards.count(card)
		matrix[index][:nums] = 1
	return matrix.flatten('F')

def encode_legal_actions(legal_actions):
	"""
	编码合法动作
	"""
	matrix = np.zeros((27, 4), dtype=np.float32)
	for legal_action in list(set(legal_actions)):
		index = new_card_encoding_dict[legal_action]
		nums = legal_actions.count(legal_action)
		matrix[index][:nums] = 1
	return matrix.flatten('F')

def encode_last_action(last_action):
	"""
	编码上一个动作
	"""
	if not last_action:
		return np.zeros(39, dtype=np.float32)
	matrix = np.zeros(39, dtype=np.float32)
	index = new_card_encoding_dict[last_action[-1]]
	matrix[index] = 1
	return matrix

def pile2list(piles):
	"""
	计算碰、杠卡牌
	"""
	cards_list = []
	for p in piles.keys():
		for pile in piles[p]:
			cards_list.extend(pile[1])
	return cards_list

def action_seq_history(action_seqs):
	"""
	编码出牌历史记录
	"""
	if len(action_seqs) == 0:
		return np.zeros(156, dtype=np.float32)
	matrix = np.zeros([4, 27], dtype=np.float32)
	# 取一轮玩家出牌动作进行编码
	counter = Counter(action_seqs[-4:])
	for card, nums in counter.items():
		matrix[:, const.Card2Column[card]] = const.NumOnes2Array[nums]
	return matrix.flatten('F')

def encode_num(hand_card_nums):
	"""
	编码卡牌数量
	"""
	matrix = np.zeros([4, 14], dtype=np.float32)
	for idx, nums in enumerate(hand_card_nums):
		matrix[idx][nums - 1] = 0
	return matrix.flatten('F')