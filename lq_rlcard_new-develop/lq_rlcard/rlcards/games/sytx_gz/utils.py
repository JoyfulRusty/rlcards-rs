# -*- coding: utf-8 -*-

from rlcards.const.sytx_gz import const
from rlcards.games.sytx_gz.card import SyCard as Card

def make_cards():
	"""
	构建卡牌(54)
	"""
	num = 0
	card_encoding_dict = {}
	for card in const.CARD_INDEX:
		card_encoding_dict[card] = num
		num += 1
	return add_actions(card_encoding_dict, num)

def add_actions(encoding_dict, num):
	"""
	添加水鱼动作类型索引
	"""
	encoding_dict[const.ActionType.SP] = num + 1
	encoding_dict[const.ActionType.QG] = num + 2
	encoding_dict[const.ActionType.MI] = num + 3
	encoding_dict[const.ActionType.ZOU] = num + 4
	encoding_dict[const.ActionType.SHA] = num + 5
	encoding_dict[const.ActionType.KAI] = num + 6
	encoding_dict[const.ActionType.REN] = num + 7
	encoding_dict[const.ActionType.XIN] = num + 8
	encoding_dict[const.ActionType.FAN] = num + 9

	return encoding_dict

def make_actions():
	"""
	构建动作序列
	"""
	num = 0
	action_encoding_dict = {}
	for card in const.ActionType:
		action_encoding_dict[card] = num
		num += 1

	return action_encoding_dict


# 构建卡牌和动作索引映射，用于解码动作
new_card_encoding_dict = make_cards()
card_decoding_dict = {new_card_encoding_dict[key]: key for key in new_card_encoding_dict.keys()}

# 构建动作索引，用于解码动作
new_action_encoding_dict = make_actions()
action_decoding_dict = {new_action_encoding_dict[key]: key for key in new_action_encoding_dict.keys()}


def init_deck():
	"""
	初始化卡牌对象
	"""
	deck = []
	index_num = 0
	for card_value in const.ALL_CARDS:
		card = Card(card_value//100, card_value)
		card.set_index_num(index_num)
		index_num += 1
		deck.append(card)

	return deck


if __name__ == "__main__":
	print(new_card_encoding_dict)
	print(init_deck())