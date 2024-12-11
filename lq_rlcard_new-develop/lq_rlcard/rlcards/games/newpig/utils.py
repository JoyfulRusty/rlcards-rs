# -*- coding: utf-8 -*-

import functools

from rlcards.const.pig import const
from rlcards.games.newpig.card import Card

def make_cards():
	"""
	构建卡牌索引序列
	"""
	num = 0
	card_encoding_dict = {}
	for card in const.ALL_POKER:
		card_encoding_dict[card] = num
		num += 1
	return card_encoding_dict

# 构建卡牌和动作索引映射，用于解码动作
new_card_encoding_dict = make_cards()
card_decoding_dict = {new_card_encoding_dict[key]: key for key in new_card_encoding_dict.keys()}

def init_deck():
	"""
	初始化卡牌对象
	"""
	deck = []
	index_num = 0
	for value in const.ALL_POKER:
		card = Card(value // 100, value)
		card.set_index_num(index_num)
		index_num += 1
		deck.append(card)
	return deck

def get_move_type(move):
	"""
	获取移动类型
	"""
	move_size = len(move)
	if move_size == 0:
		return {'type': const.TYPE_0_PASS}
	return {'type': const.TYPE_1_SINGLE, 'suit': move[0] // 100}

def lp_policy(cards, yet_lp):
	"""
	亮牌策略
	"""
	light_cards = []
	can_lp = list(set(cards).intersection(const.CAN_LIGHT_CARDS))
	if not can_lp:
		return light_cards
	if not cards or not isinstance(cards, list):
		return light_cards
	suit_cards = dict()
	for card in cards:
		suit = int(card / 100)
		suit_cards.setdefault(suit, []).append(card)
	cards_dis = get_cards_distribution(suit_cards)
	for cl in can_lp:
		suit = int(cl / 100)
		type_cards = suit_cards.get(suit)
		if cl == const.ZHU:
			if len(type_cards) > 4:
				light_cards.append(cl)
		elif cl == const.YANG:
			if len(type_cards) > 4:
				if cards_dis != [3, 3, 4, 4]:
					light_cards.append(cl)
		elif cl == const.BAN:
			if const.ZHU not in yet_lp and const.YANG not in yet_lp:
				if len(type_cards) > 4:
					light_cards.append(cl)
		elif cl == const.KING:
			if len(type_cards) > 4:
				if len(gt_or_lt_xx_cards(type_cards, const.HX_J)) > 4:
					light_cards.append(cl)
				elif 315 not in type_cards:
					if cal_xx_cards_num(cards_dis) > 0:
						light_cards.append(cl)
	return light_cards

def get_cards_distribution(suit_cards):
	"""
	根据卡牌花色数量排序
	"""
	cards_dis = []
	if not suit_cards:
		return cards_dis
	for sc in suit_cards:
		cards_dis.append(len(suit_cards.get(sc, [])))
	cards_dis.sort()
	return cards_dis

def gt_or_lt_xx_cards(cards, xx, lt=False):
	"""
	大于或小于xx牌
	"""
	gt_cards = []
	for card in cards:
		if not lt and card >= xx:
			gt_cards.append(card)
		if lt and card <= xx:
			gt_cards.append(card)
	return gt_cards

def cal_xx_cards_num(cards, num=2):
	"""
	分区牌数小于2的
	"""
	count = 0
	for card in cards:
		if card <= num:
			count += 1
	return count

if __name__ == '__main__':
	hand = [107, 110, 114, 203, 305, 306, 309, 313, 316, 406, 407, 410, 412]
	res = lp_policy(hand, [])
	print(res)