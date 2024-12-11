# -*- coding: utf-8 -*-

# todo: 配牌

import itertools
import numpy as np

from rlcards.games.sytx.rules import Rules
from rlcards.games.sytx import config


def random_choice_card(card_arr, p=None):
	"""
	有放回抽样
	"""
	if not p:
		p = []
	if not isinstance(p, list) or not len(card_arr):
		return
	if p:
		extra_num = len(card_arr) - len(p)
		if extra_num > 0:
			p.extend([0 for _ in range(extra_num)])
		elif extra_num < 0:
			for _ in range(-1 * extra_num):
				p.pop()
	if sum(p) != 1:
		p = None
	return np.random.choice(a=card_arr, size=1, replace=True, p=p)[0]

def special_pair_pp(pair_sum, max_three_index, include_pair_index, all_combine, is_landlord):
	"""
	5对、9对时，采取下列策略配牌
	"""
	max_three_type_index = np.append(max_three_index, include_pair_index)
	if pair_sum == 10:
		better_combine = all_combine[include_pair_index, ::, ::]
	elif pair_sum == 18:
		p = config.p14 if is_landlord else config.p15
		best_combine_index = random_choice_card(max_three_type_index, p)
		better_combine = all_combine[best_combine_index, ::, ::]
	else:
		p = config.p16 if is_landlord else config.p17
		best_combine_index = random_choice_card(max_three_type_index, p)
		better_combine = all_combine[best_combine_index, ::, ::]
	return better_combine

def is_pair_and_two_max(
		cards_groups_arr,
		max_three_index,
		include_pair_index,
		include_pair_index1,
		two_pu_sum,
		all_combine,
		is_landlord):
	"""
	对子并计算两张最大
	"""
	pair_sum = sum(cards_groups_arr[include_pair_index, include_pair_index1, :])
	# 如果不配豹子的点数相加 > 14，并且豹子点数在J以下，可将权重设置高一点
	if pair_sum <= 2:
		better_combine = all_combine[include_pair_index, ::, ::]
	elif max(two_pu_sum) > 15:  # 最少是 7Q 7K这种牌型  55 34  22 57
		better_combine = special_pair_pp(pair_sum, max_three_index, include_pair_index, all_combine, is_landlord)
	else:
		better_combine = all_combine[include_pair_index, ::, ::]
	return better_combine

def pei_pai_by_a_pair(cards_groups_arr, max_three_index, two_pu_sum, all_combine, is_landlord):
	"""
	配对豹子
	"""
	include_pair_index = np.argwhere(cards_groups_arr[:, :, 0] == cards_groups_arr[:, :, 1])[0, 0]
	include_pair_index1 = np.argwhere(cards_groups_arr[:, :, 0] == cards_groups_arr[:, :, 1])[0, 1]
	# 当有对的组合大时只有一铺最大，当不组成对的组合时有两铺最大 todo:加上对A的情况
	if len(max_three_index) == 1:
		better_combine = all_combine[include_pair_index, ::, ::]
	elif len(max_three_index) == 2:
		# todo:对子和点数都需要考虑
		better_combine = is_pair_and_two_max(
			cards_groups_arr,
			max_three_index,
			include_pair_index,
			include_pair_index1,
			two_pu_sum,
			all_combine,
			is_landlord
		)
	else:
		better_combine = all_combine[include_pair_index, ::, ::]
	return better_combine.flatten()

def combine_card(cards: list):
	"""
	组合卡牌
	"""
	card_groups = list()
	all_combine = list(itertools.combinations(cards, 2))
	for _ in range(3):
		card_groups.append([all_combine.pop(0), all_combine.pop(-1)])
	return card_groups

def get_card_groups_arr(card):
	"""
	获取卡牌组属性
	"""
	return np.array(combine_card(card))

def pei_pai_forward(cards):
	"""
	配牌方向
	"""
	v_count_dict, card_v_list = Rules.get_count_list_by_value(cards)
	cards_v = Rules.abstract_value_to_dan_zhang(card_v_list)
	cards_v_arr = get_card_groups_arr(cards_v)
	# per_pu_sum = cards_v_arr.sum(2) % 10
	# two_pu_sum = per_pu_sum.sum(1)
	per_pu_sum = np.round(cards_v_arr.sum(2) % 10, decimals=1)
	two_pu_sum = np.round(per_pu_sum.sum(1), decimals=1)
	max_three_combine_index = np.where(two_pu_sum == max(two_pu_sum))[0]
	return v_count_dict, max_three_combine_index, per_pu_sum, two_pu_sum, cards_v_arr

def get_pro_two_big(is_landlord, is_first=True):
	"""
	组合中两铺大且含9
	"""
	if is_landlord:
		if is_first:
			p = [0.9, 0.1]  # 庄保大
		else:
			p = [0.1, 0.9]  # 庄保大
	else:
		if is_first:
			p = [0.7, 0.3]
		else:
			p = [0.3, 0.7]
	return p

def cal_card_distance(card1, card2, card3=None, combine_num=2):
	"""
	计算卡牌距离，也就是相差的值
	"""
	if combine_num == 2:
		diff1 = abs(card1[0] - card1[1])
		diff2 = abs(card2[0] - card2[1])
		return diff1, diff2
	elif combine_num == 3:
		diff1 = abs(card1[0] - card1[1])
		diff2 = abs(card2[0] - card2[1])
		diff3 = abs(card3[0] - card3[1])
		return diff1, diff2, diff3

def get_index_exist_nine(nine_row, max_three_index, is_landlord):
	"""
	当有9且有两个组合大时
	"""
	if nine_row == max_three_index[0]:
		p = get_pro_two_big(is_landlord)
		best_combine_index = random_choice_card(max_three_index, p)  # todo: 写在配置中
	else:
		p = get_pro_two_big(is_landlord, is_first=False)
		best_combine_index = random_choice_card(max_three_index, p)
	return best_combine_index

def one_nine_two_max(include_nine_index, all_combine, per_pu_sum, max_three_index, is_landlord):
	"""
	只有一个9，但有两个组合大
	"""
	nine_row = include_nine_index[0][0]
	nine_column = include_nine_index[0][1]
	nine_card = all_combine[nine_row][nine_column]
	if Rules.value(nine_card[0]) == 1:
		if sum(per_pu_sum[max_three_index[0]]) >= 15:
			best_combine_index = get_index_exist_nine(nine_row, max_three_index, is_landlord)
		else:
			best_combine_index = nine_row
	else:
		best_combine_index = get_index_exist_nine(nine_row, max_three_index, is_landlord)
	return best_combine_index

def is_two_max(max_three_index, per_pu_sum, all_combine, is_landlord):
	"""
	存在2个组合之和最大时
	"""
	try:
		include_nine_index = np.argwhere(per_pu_sum >= 9)  # todo:2个9时
		if len(include_nine_index) == 2:
			best_combine_index = random_choice_card(max_three_index, [0.7, 0.3])
		else:
			best_combine_index = one_nine_two_max(include_nine_index, all_combine, per_pu_sum, max_three_index, is_landlord)
		better_combine = all_combine[best_combine_index, ::, ::]
	except IndexError:
		diff1, diff2 = cal_card_distance(per_pu_sum[max_three_index[0], :], per_pu_sum[max_three_index[1], :], 2)
		if abs(diff1 - diff2) < 1:  # 两张花牌的情况
			if diff1 > diff2:
				p = [0.7, 0.3]
			else:
				p = [0.3, 0.7]
			rc = random_choice_card(max_three_index, p)
			better_combine = all_combine[rc, :, :]
		elif diff1 > diff2:
			better_combine = all_combine[max_three_index[1], :, :]
		else:
			better_combine = all_combine[max_three_index[0], :, :]
	return better_combine

def get_p_by_diff(diffs, d):
	"""
	获取距离
	"""
	if min(diffs) <= d:
		p = [0.7, 0.3]
	else:
		p = [0.9, 0.1]
	return p

def is_three_max(max_three_index, per_pu_sum, two_pu_sum, all_combine):
	"""
	3个组合之和都相等时
	"""
	diff1, diff2, diff3 = cal_card_distance(
		per_pu_sum[max_three_index[0], :],
		per_pu_sum[max_three_index[1], :],
		per_pu_sum[max_three_index[2], :], 3
	)
	diffs = [diff1, diff2, diff3]
	try:
		include_nine_index = np.argwhere(per_pu_sum >= 9)[0][0]  # 带9的组合索引
		min_diff_index = diffs.index(min(diffs))
		best_two_combine = np.array([include_nine_index, min_diff_index])
		if two_pu_sum[0] < 10:  # 判断大小铺相加是否大于10
			p = get_p_by_diff(diffs, 1)
		else:
			p = get_p_by_diff(diffs, 2)
		best_combine_index = random_choice_card(best_two_combine, p)
		better_combine = all_combine[best_combine_index, ::, ::]
	except IndexError:
		if abs(diff2 - diff3) < 1:
			best_index = random_choice_card([1, 2], p=[0.6, 0.4])
		else:
			best_index = diffs.index(min(diffs))
		better_combine = all_combine[best_index, ::, ::]
	return better_combine

def pei_pai_by_dz(max_three_index, per_pu_sum, two_pu_sum, all_combine, is_landlord):
	"""
	癞子牌存在时，配牌
	"""
	if len(max_three_index) == 1:
		max_combine_index = two_pu_sum.argmax()
		better_combine = all_combine[max_combine_index, ::, ::]
	elif len(max_three_index) == 2:
		better_combine = is_two_max(max_three_index, per_pu_sum, all_combine, is_landlord)
	else:
		better_combine = is_three_max(max_three_index, per_pu_sum, two_pu_sum, all_combine)
	return better_combine.flatten()

def pei_pai(cards, all_combine, is_landlord):
	"""
	配牌
	"""
	v_count_dict, max_three_combine_index, per_pu_sum, two_pu_sum, cards_v_arr = pei_pai_forward(cards)
	if len(v_count_dict) == 4:  # 单张
		return pei_pai_by_dz(max_three_combine_index, per_pu_sum, two_pu_sum, all_combine, is_landlord)
	if len(v_count_dict) == 3 or 2:
		return pei_pai_by_a_pair(cards_v_arr, max_three_combine_index, two_pu_sum, all_combine, is_landlord)

def sort_val(card):
	"""
	值排序
	"""
	return card % 100, card // 100

def pei_pai_by_lai_zi(cards, cards_copy, all_combine, is_landlord):
	"""
	癞子版配牌
	"""
	with_lai_zi = Rules.lz_cards()
	p_lai_zi = Rules.check_has_lai_zi(cards_copy)
	if not p_lai_zi:
		combined_card = pei_pai(cards_copy, all_combine, is_landlord)
		res = map(int, combined_card)
		return list(res)
	for n in p_lai_zi:
		cards_copy.remove(n)
	cards_copy.sort(key=sort_val)
	res, cards_v_list = Rules.get_count_list_by_value(cards_copy)
	val_len = len(res)
	if len(p_lai_zi) == len(with_lai_zi):
		if val_len == 1:
			cards.sort(reverse=True, key=Rules.value)
			return cards
		else:
			if cards_v_list[0] != 1:
				if cards_v_list[0] > cards_v_list[1]:
					return [with_lai_zi[-1], cards_copy[0], with_lai_zi[0], cards_copy[-1]]
				return [with_lai_zi[-1], cards_copy[1], with_lai_zi[0], cards_copy[0]]
			return [with_lai_zi[-1], cards_copy[0], with_lai_zi[0], cards_copy[-1]]
	else:
		if val_len == 1:
			cards.sort(reverse=True)
			return cards
		elif val_len == 2:
			# 匹配水鱼
			if cards_v_list[0] == cards_v_list[1]:
				return [p_lai_zi[0], cards_copy[-1], cards_copy[0], cards_copy[1]]
			return [p_lai_zi[0], cards_copy[0], cards_copy[1], cards_copy[-1]]

		# 对子 + 点数
		cards_v = Rules.abstract_value_to_dan_zhang(cards_v_list)  # 0 < 花牌 < 1
		all_comb = list(itertools.combinations(cards_copy, 2))
		all_val_combine = list(itertools.combinations(cards_v, 2))
		comb_arr = np.array(all_val_combine)
		per_sum = comb_arr.sum(1) % 10  # 每个组合之和
		if per_sum.max() < config.THRESHOLD_NUM:
			if cards_v_list[0] != 1:
				cards_copy.extend(p_lai_zi)  # 紧最大的对子扯
				return cards_copy
			p_lai_zi.extend(cards_copy)
			return p_lai_zi
		max_index = per_sum.argmax()
		better = all_comb[max_index]
		res_card = set(cards_copy).difference(better)
		p_lai_zi.append(res_card.pop())
		p_lai_zi.extend(better)
		return p_lai_zi

def get_better_combine(cards: list, is_landlord=False):
	"""
	对普通牌进行配牌
	"""
	if not cards or len(cards) != 4:
		return
	with_lai_zi = Rules.check_has_lai_zi(cards)
	cards_copy = cards[:]
	cards_copy.sort(key=Rules.value)
	all_combine = get_card_groups_arr(cards_copy)
	if with_lai_zi:
		return pei_pai_by_lai_zi(cards, cards_copy, all_combine, is_landlord)
	combined_card = pei_pai(cards_copy, all_combine, is_landlord)
	res = map(int, combined_card)
	return list(res)