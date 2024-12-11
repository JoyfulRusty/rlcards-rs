# -*- coding: utf-8 -*-

import pickle
import numpy as np

from itertools import product
from rlcards.games.mahjong.dfs_xts.utils import *

# 计算胡牌组合
""" compose_list = [
    [(0, 4), (0, 0), (0, 0), (0, 0)],
    [(0, 3), (0, 0), (0, 0), (0, 1)],
    ......
    [(0, 0), (4, 0), (0, 0), (0, 0)]
    ]
    [万, 条, 筒, 字牌] -> 万: (顺子数, 刻子数)
    len(compose_list) = 210
"""

compose_list = []  # length = 210
for x in range(5):
	compose_list_sz = compose_gen_sz(x)
	compose_list_kz = compose_gen_kz(4 - x)
	for c1 in compose_list_sz:
		for c2 in compose_list_kz:
			compose_row = []
			for x, y in zip(c1, c2):
				compose_row.append((x, y))
			# 字牌
			compose_row.append((0, c2[3]))
			compose_list.append(compose_row)

# 计算三花色下顺子与刻子组合片段，二维数组
"""
fragment_list[顺子个数][刻子个数]
fragment_list = [[...],[...],[...],[...],[...]],
                [[...],[...],[...],[...]],
                [[...],[...],[...]],
                [[...],[...]],
                [[...],]]
[0,01234]
[1,0123]
[2,012]
[3,01]
[4,0]
"""

fragment_list = [[], [], [], [], []]  # 片段列表
one_sz_list = [produce_sz_9(x) for x in range(0, 7)]
one_kz_list = [produce_kz_9(x) for x in range(0, 9)]
for x in range(5):
	for y in range(5 - x):
		sz_kz_list_to_product = [one_sz_list for xi in range(x)] + [one_kz_list for yi in range(y)]
		result = product(*sz_kz_list_to_product)
		""" 
		一顺三刻
		r = (
			[0, 0, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 3, 0, 0],
			[0, 0, 0, 0, 0, 0, 3, 0, 0],
			[0, 0, 3, 0, 0, 0, 0, 0, 0]
			)

		一顺一刻
		r = ([0, 0, 1, 1, 1, 0, 0, 0, 0],
			 [3, 0, 0, 0, 0, 0, 0, 0, 0])
		"""
		y_list = []
		for r in result:
			if not r:
				break
			r_np = np.array(r)
			r_np = r_np.sum(axis=0)
			if r_np.max() < 5:
				try:
					y_list = np.vstack((y_list, r_np))
				except ValueError:
					y_list = r_np
		y_list = np.unique(y_list, axis=0)
		if y_list.size <= 0:
			y_list = np.array([[0] * 9])
		fragment_list[x].append(y_list)

# 计算字牌下，不同刻子数的组合片段，一维数组
# fragment_list_zp[刻子个数]

fragment_list_zp = []
# 一个刻子可能形成字牌[7]
one_kz_list = [produce_kz_7(x) for x in range(0, 7)]
for x in range(5):
	kz_list_to_product = [one_kz_list for xi in range(x)]
	result = product(*kz_list_to_product)
	y_list = []
	for r in result:
		if not r:
			break
		r_np = np.array(r)
		r_np = r_np.sum(axis=0)
		if r_np.max() < 5:
			try:
				y_list = np.vstack((y_list, r_np))
			except ValueError:
				y_list = r_np
	y_list = np.unique(y_list, axis=0)
	if y_list.size <= 0:
		y_list = np.array([[0] * 7])
	fragment_list_zp.append(y_list)

res_by_ron_set = set()
# 一般形式的计算
k = 0
for compose in compose_list:
	k += 1
	if k % 5 == 0:
		print(f"ron_set计算中，进度为: ({k} / {len(compose_list)}))")
	my_list = []
	# 三花色
	for x in compose[0: 3]:
		r_np = fragment_list[x[0]][x[1]]
		my_list.append(r_np)
	# 字牌
	x = compose[3]
	r_np = fragment_list_zp[x[1]]
	my_list.append(r_np)
	product_list = list(product(*my_list))
	for p in product_list:
		p_concatenate = np.concatenate(p, axis=0)
		# hand_card_list.append(p_concatenate.tolist())
		for x in range(len(p_concatenate)):
			p_concatenate[x] += 2
			if p_concatenate[x] < 4:
				# 编码
				ehc = encode_hand_cards(p_concatenate)
				res_by_ron_set.add(ehc)
			p_concatenate[x] -= 2
print("计算完成")

# 保存
print("正在写入 ron_set.pickle.....")
with open("ron_set.pickle", "wb") as f:
	pickle.dump(res_by_ron_set, f)
print("写入完成.")