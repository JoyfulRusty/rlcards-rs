# -*- coding: utf-8 -*-

import threading
import collections

from rlcards.const.doudizhu import const as CONST
from rlcards.games.doudizhu.card import Card
from rlcards.data.doudizhu.read_data import CARD_TYPE_DATA, TYPE_CARD_DATA

def init_54_deck():
	"""
	初始化一副标准的 52 张牌，BJ 和 RJ
	:return: (list)卡片对象列表
	"""
	suit_list = ['S', 'H', 'D', 'C']
	rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
	res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
	res.append(Card('BJ', ''))
	res.append(Card('RJ', ''))
	return res

def ddz_sort_str(card_1, card_2):
	"""
	比较str表示的两张牌大小
	:param card_1: 一张str表示的单牌
	:param card_2: 一张str表示的单牌
	:return: int: 1(card_1 > card_2) / 0(card_1 = card2) / -1(card_1 < card_2)
	"""
	key_1 = CONST.CARD_RANK_STR.index(card_1)
	key_2 = CONST.CARD_RANK_STR.index(card_2)
	if key_1 > key_2:
		return 1
	if key_1 < key_2:
		return -1
	return 0

def ddz_sort_card_obj(card_1, card_2):
	"""
	比较Card对象的两张卡牌的大小
	:param card_1: 卡牌对象1
	:param card_2: 卡牌对象2
	"""
	key = []
	for card in [card_1, card_2]:
		if card.rank == '':
			key.append(CONST.CARD_RANK.index(card.suit))
		else:
			key.append(CONST.CARD_RANK.index(card.rank))
	if key[0] > key[1]:
		return 1
	if key[0] < key[1]:
		return -1
	return 0

def get_landlord_score(curr_hand_cards):
	score_map = {'A': 1, '2': 2, 'B': 3, 'R': 4}
	score = 0

	# 鬼炸
	if curr_hand_cards[-2: ] == 'BR':
		score += 8
		curr_hand_cards = curr_hand_cards[:-2]

	length = len(curr_hand_cards)
	i = 0
	while i < length:
		# bomb
		if i <= (length - 4) and curr_hand_cards[i] == curr_hand_cards[i + 3]:
			score += 6
			i += 4
			continue
		# black joker, red block
		if curr_hand_cards[i] in score_map:
			score += score_map[curr_hand_cards[i]]
		i += 1
	return score

def cards2str_with_suit(cards):
	"""
	获取带花色卡牌对应的字符串表示
	:param cards(list): 卡牌对象列表
	:return: string卡牌的字符串表示
	"""
	return ''.join([card.suit + card.rank for card in cards])


def cards2str(cards):
	"""
	获取带花色的牌对应的字符串表示
	:param cards(list): 卡牌对象列表
	:return: (string)卡牌的字符串表示
	"""
	response = ''
	for card in cards:
		if card.rank == '':
			response += card.suit[0]
		else:
			response += card.rank
	return response

class LocalObjs(threading.local):
	def __init__(self):
		self.cached_candidate_cards = None
_local_objs = LocalObjs()


def contains_cards(candidate, target):
	"""
	查找候选人的卡牌中是否包含目标卡牌
	:param candidate(string): 代表候选人卡牌的字符串
	:param target(string): 代表目标卡牌张数的字符串
	:return: True / False
	"""
	# 正常情况下，大多数连续调用这个函数，将针对用以候选人不同的目标
	# 所以candidate中每张卡牌的缓存计数可以加快，
	# 如果候选人保持不变，则进行后续测试的比较
	if not _local_objs.cached_candidate_cards or _local_objs.cached_candidate_cards != candidate:
		_local_objs.cached_candidate_cards = candidate
		cards_dict = collections.defaultdict(int)
		for card in candidate:
			cards_dict[card] += 1
		_local_objs.cached_candidate_cards_dict = cards_dict
	cards_dict = _local_objs.cached_candidate_cards_dict
	if (target == ''):
		return True
	curr_card = target[0]
	curr_count = 1
	for card in target[1:]:
		if (card != curr_card):
			if (cards_dict[curr_card] < curr_count):
				return False
			curr_card = card
			curr_count = 1
		else:
			curr_count += 1
	if (cards_dict[curr_card] < curr_count):
		return False
	return True

def encode_cards(plane, cards):
	"""
	对卡牌进行编码，并将其重新保存到平面中
	:param plane: 平面
	:param cards: 卡牌列表，每个条目都是卡牌独特的特征
	"""
	if not cards:
		return None

	layer = 1
	if len(cards) == 1:
		rank = CONST.CARD_RANK_STR.index(cards[0])
		plane[layer][rank] = 1
		plane[0][rank] = 0

	else:
		for index, card in enumerate(cards):
			if index == 0:
				continue
			if card == cards[index - 1]:
				layer += 1
			else:
				rank = CONST.CARD_RANK_STR.index(cards[index - 1])
				plane[layer][rank] = 1
				layer = 1
				plane[0][rank] = 0

		rank = CONST.CARD_RANK_STR.index(cards[-1])
		plane[layer][rank] = 1
		plane[0][rank] = 0

def get_gt_card(player, greater_player):
	"""
	提供比玩家打出的牌大的玩家，一轮中的前一位瓦加
	:param player(obj): 等待出牌的玩家
	:param greater_player(obj): 当前打出最大牌的玩家
	:return: (list)一串更大的卡牌列表
	note: 返回值，包含pass
	"""
	# add pass to legal actions
	gt_cards = ['pass']
	current_hand = cards2str(player.current_hand)
	target_cards = greater_player.played_cards  # 上一个玩家打出的最大
	target_types = CARD_TYPE_DATA[0][target_cards]
	type_dict = {}

	for card_type, weight in target_types:
		if card_type not in type_dict:
			type_dict[card_type] = weight

	if 'rocket' in type_dict:
		return gt_cards
	type_dict['rocket'] = -1
	if 'bomb' not in type_dict:
		type_dict['bomb'] = -1

	for card_type, weight in type_dict.items():
		candidate = TYPE_CARD_DATA[card_type]
		for can_weight, cards_list in candidate.items():
			if int(can_weight) > int(weight):
				for cards in cards_list:
					# TODO: improve efficiency
					if cards not in gt_cards and contains_cards(current_hand, cards):
						# if self.contains_cards(current_hand, cards):
						gt_cards.append(cards)
	return gt_cards