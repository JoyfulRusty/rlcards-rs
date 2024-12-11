# -*- coding: utf-8 -*-

import pickle

from rlcards.games.mahjong.dfs_xts.dfs import *
from rlcards.games.mahjong.dfs_xts.utils import *
from rlcards.utils import cal_time


def check_hu(hc: str):
	"""
	检查胡
	"""
	hc = convert_hc_to_list(hc)
	if sum(hc) != 14:
		raise ValueError("请传入14张卡牌")
	with open("ron_set.pickle", 'rb') as f:
		ron_set = pickle.load(f)
	ehc = encode_hand_cards(hc)
	if ehc in ron_set:
		print("can hu")
	else:
		print("not can hu")

def calc_hu_13(hc=None, hc_list=None):
	"""
	计算13张卡牌胡
	"""
	if hc_list:
		hc = hc_list
	else:
		hc = convert_hc_to_list(hc)
	if sum(hc) != 13:
		raise ValueError("请传入13张卡牌")
	m = get_mz_combine(hc)
	# 无面子拆解情况，传入空数组
	if not m:
		m = [[]]
	# 最大向听数为8
	xt_list = [[]] * 8
	for x in m:
		# 面子数量
		mz_count = len(x)
		# 减去面子牌
		thc = get_sub_mz_hc(hc.copy(), x)
		# 计算剩余的搭子牌 -> (index, 对子，二连顺，间隔顺)
		dz_list = get_dz_combine(thc)
		dz_list = get_min_dz_list(dz_list)
		dz_list_xt_min = 8
		for dz in dz_list:
			# 计算雀头
			if_que = 0
			if dz[1] > 0:
				if_que = 1
			dz_count = len(dz_list)
			xt = calc_xts(mz_count, dz_count, if_que)
			if xt < dz_list_xt_min:
				# 减去搭子数
				dz_hc = get_sub_dz_hc(thc.copy(), dz_list)
				# 获取剩余单张卡牌
				single_list = get_dan_zhang(dz_hc)
				# 进张
				tp_ting = get_tp_from_dz(dz_list, xt)

				# todo: 或许存在更多种情况
				# 向听数为0
				if xt == 0:
					# 无搭子，即单吊
					if not dz_list:
						tp_ting += single_list
				# 向听数为1
				if xt == 1:
					if dz_count == 1:
						if if_que:
							single_around = get_dan_zhang_around(single_list)
							tp_ting += single_around
							tp_ting += single_list
						else:
							tp_ting += single_list
					if dz_count == 2:
						# 搭子自身可减少向听
						for dz in dz_list:
							i = dz[0]
							if dz[1] > 0:
								tp_ting.append(i)
							elif dz[2] > 0:
								tp_ting.append(i)
								tp_ting.append(i + 1)
							elif dz[3] > 0:
								tp_ting.append(i)
								tp_ting.append(i + 2)
				# 向听数为2以上
				if xt >= 2:
					if mz_count + dz_count < 5:
						# 4搭子 + 0雀头，不需要新搭子(顺子型)
						if mz_count + dz_count == 4 and not if_que:
							# 有效进张卡牌
							less_than5 = get_md_less_than5(dz_hc, 0)
							tp_ting += less_than5
						else:
							# 有效进张卡牌
							less_than5 = get_md_less_than5(dz_hc)
							tp_ting += less_than5
					elif mz_count + dz_count >= 5:
						# 超载时，搭子自身可化为雀头，单张也可
						if not if_que:
							for dz in dz_list:
								i = dz[0]
								if dz[1] > 0:
									tp_ting.append(i)
								elif dz[2] > 0:
									tp_ting.append(i)
									tp_ting.append(i + 1)
								elif dz[3] > 0:
									tp_ting.append(i)
									tp_ting.append(i + 2)
							tp_ting += single_list
				tp_ting = list(set(tp_ting))
				tp_ting.sort()
				xt_list[xt] += tp_ting

	# 向听数及进牌列表
	for y in range(len(xt_list)):
		if xt_list[y]:
			# [向听数，进张列表]
			return (y, list(set(xt_list[y])))

def get_min_dz_list(dz_list: list):
	"""
	获取最小的搭子组合
	"""
	mm_list = min(dz_list, key=len)
	return mm_list

def calc_hu_14(hc: str):
	"""
	14张牌理分析
	"""
	hc = convert_hc_to_list(hc)
	if sum(hc) != 14:
		raise ValueError("请传入14张卡牌")
	xt_list = []
	# todo: 计算对应出牌及进张卡牌
	for x in range(len(hc)):
		if hc[x] > 0:
			# 换位
			hc[x] -= 1
			xt = calc_hu_13(hc_list=hc)
			if xt:
				xt_list.append([x, xt])
			# 复位
			hc[x] += 1

	# todo: 最小向听数
	xt_min = min([x[1][0] for x in xt_list])
	if xt_min == 0:
		print("听牌")
	else:
		print(f"向听数为: {xt_min}\n")

	card_advice_list = []
	for xxt in xt_list:
		xt = xxt[1]
		if xt[0] == xt_min:
			xt[1].sort()
			my_sum = calc_tp_sum(hc, xt[1])
			card_advice_list.append([xxt[0], xt[1], my_sum])
	card_advice_list.sort(key=lambda x: x[2], reverse=1)
	for x in card_advice_list:
		deal_x = list(set(x[1]))
		res = convert_num_to_card(x[0])
		if res is None:
			continue
		print(f"打{res} <==>: 进张卡牌及数量:" ,[convert_num_to_card(x) for x in deal_x if convert_num_to_card(x) is not None], f"{len(deal_x)}枚\n")

@cal_time
def main():
	calc_hu_14('12368c12372b135d1f')

if __name__ == "__main__":
	main()