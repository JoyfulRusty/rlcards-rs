# -*- coding: utf-8 -*-

import re
import math

def compose_gen_sz(sz) -> list:
	"""
	指定数量的顺子在三花色中组合
	"""
	my_compose = []

	def my_recursion(fr=None, br=None):
		"""
		递归
		"""
		if fr is not None and br is not None:
			my_compose.append([sz - fr, br, fr - br])
			return
		else:
			for k in range(0, fr + 1):
				my_recursion(fr, k)
	for x in range(sz + 1):
		# fr = sz - x
		my_recursion(x)

	return my_compose

def compose_gen_kz(kz) -> list:
	"""
	指定数量刻子在三花色及字牌中的组合
	"""
	my_compose = []

	def my_recursion(fr=None, br=None, cr=None):
		if fr is not None and br is not None and cr is not None:
			my_compose.append([kz - fr, br, cr, fr - br - cr])
			return
		elif fr is not None and br is not None:
			for k in range(0, fr - br + 1):
				my_recursion(fr, br, k)
		else:
			for k in range(0, fr + 1):
				my_recursion(fr, k)
	for x in range(kz + 1):
		# fr = kz - x
		my_recursion(x)
	return my_compose

def produce_kz_7(index) -> list:
	"""
	产生七位花色的刻子，字牌
	"""
	my_kz = [0] * 7
	my_kz[index] += 3
	return my_kz

def produce_kz_9(index) -> list:
	"""
	产生9为花色的刻子
	"""
	my_kz = [0] * 9
	my_kz[index] += 3
	return my_kz

def produce_sz_9(index) -> list:
	"""
	产生9位花色的顺子
	"""
	my_sz = [0] * 9
	my_sz[index] += 1
	my_sz[index + 1] += 1
	my_sz[index + 2] += 1
	return my_sz

def encode_hand_cards(hc: list) -> str:
	"""
	编码手牌
	"""
	def encode_hc(ehc, zp=False):
		"""
		编码
		"""
		if zp:
			mhc = "0".join(str(y) for y in ehc)
		else:
			mhc = "".join(str(y) for y in ehc)
		mhc = mhc.strip("0")
		mhc = re.sub(r"0{2,}", "0", mhc)
		return mhc

	hc_chars = hc[0:9]  # 万
	hc_bamboos = hc[9:18]  # 条
	hc_dots = hc[18:27]  # 筒
	hc_zi = hc[27:34]  # 字

	ehc_to_join = []
	for x in [
		encode_hc(hc_chars),
		encode_hc(hc_bamboos),
		encode_hc(hc_dots),
		encode_hc(hc_zi, zp=True),
	]:
		if x:
			ehc_to_join.append(x)
	return "0".join(ehc_to_join)

def convert_hc_to_list(hc: str) -> list:
	"""
	todo: 将字符串转为列表
	"""
	if not hc:
		raise ValueError
	# 赤宝牌处理
	hc = hc.replace("0", "5")
	hc_list = [0] * 28
	pattern = re.compile(r"\d+[cbdf]")  # 万条筒发
	result = pattern.findall(hc)
	for x in result:
		# todo: 编码卡牌位置 0 -> 1 or +=1
		if x[-1] == "c":  # 万
			for y in x[:-1]:
				hc_list[int(y) - 1] += 1
		if x[-1] == "b":  # 条
			for y in x[:-1]:
				hc_list[int(y) - 1 + 9] += 1
		if x[-1] == "d":  # 筒
			for y in x[:-1]:
				hc_list[int(y) - 1 + 18] += 1
		if x[-1] == "f":  # 万能牌
			for y in x[:-1]:
				hc_list[int(y) - 1 + 27] += 1
	return hc_list

def encode_arbitrary_cards(hc: list):
	"""
	不考虑花色边界和牌数，编码手牌
	"""
	def encode_hc(ehc, if_zi=False):
		"""
		编码
		"""
		if if_zi:
			mhc = "0".join(str(y) for y in ehc)
		else:
			mhc = "".join(str(y) for y in ehc)
		mhc = mhc.strip("0")
		mhc = re.sub(r"0{2,}", "0", mhc)
		return mhc

	return encode_hc(hc)

def convert_num_to_card(num: list):
	"""
	根据数字返回牌名
	"""
	m_card = None
	if num < 9:
		m_card = str(num + 1) + "万"
	elif 9 <= num < 19:
		m_card = str(num - 9 + 1) + "条"
	elif 18 <= num < 27:
		m_card = str(num - 18 + 1) + "筒"
	return m_card

def get_sub_mz_hc(hc, mz):
	"""
	从手牌减去面子牌
	"""
	for x in mz:
		i = x[0]
		if x[1] > 0:
			hc[i] -= 1
			hc[i + 1] -= 1
			hc[i + 2] -= 1
		elif x[2] > 0:
			hc[i] -= 3
	return hc

def get_sub_dz_hc(hc, dz):
	"""
	从手牌减去搭子牌
	"""
	for x in dz:
		i = x[0]
		if x[1] > 0:
			hc[i] -= 2
		elif x[2] > 0:
			hc[i] -= 1
			hc[i + 1] -= 1
		elif x[3] > 0:
			hc[i] -= 1
			hc[i + 2] -= 1
	return hc

def get_dan_zhang(hc):
	"""
	获取单张
	"""
	dan_zhang_list = []
	for x in range(len(hc)):
		if hc[x] == 1:
			dan_zhang_list.append(x)
	return dan_zhang_list

def get_dan_zhang_around(dz_list: list):
	"""
	获取单张附近能够组成搭子的牌
	"""
	g = []
	for x in dz_list:
		if x < 27:
			for y in [x - 2, x - 1, x + 1, x + 2]:
				if x >= 0 and math.floor(y / 9) == math.floor(x / 9):
					g.append(y)
	return g

def get_md_less_than5(hc, new_dz=1):
	"""
	m + d <= 5时，减少向听数的进张
	"""
	dan_zhang_list = []
	for x in range(len(hc)):
		if hc[x] == 1:
			dan_zhang_list.append(x)
			if x < 27 and new_dz:
				for y in [x - 2, x - 1, x + 1, x + 2]:
					if y >= 0 and math.floor(y / 9) == math.floor(x / 9):
						dan_zhang_list.append(y)
	return dan_zhang_list

def get_tp_from_dz(dz, xt):
	"""
	根据搭子和当前向听数，返回能够减少向听数的牌
	"""
	# dz = [(2, 1, 0, 0), (18, 0, 0, 1)]
	tp_list = []
	# 已听牌情况
	if xt == 0:
		if len(dz) == 2:
			if dz[0][1] > 0 and dz[1][1] > 0:
				tp_list.append(dz[0][0])
				tp_list.append(dz[1][0])
			else:
				for x in dz:
					index = x[0]
					# [11]
					if x[2] > 0:
						if index in [0, 9, 18]:
							tp_list.append(index + 2)
						elif index in [7, 16, 25]:
							tp_list.append(index - 1)
						else:
							tp_list.append(index - 1)
							tp_list.append(index + 2)
					# [101]
					if x[3] > 0:
						tp_list.append(index + 1)
		return tp_list

	# 1向听及以上
	for x in dz:
		index = x[0]
		# todo: 对子
		# [2]
		if x[1] > 0:
			tp_list.append(x[0])

		# todo: 二连顺
		# [11]
		if x[2] > 0:
			if index in [0, 9, 18]:
				tp_list.append(index + 2)
			elif index in [7, 16, 25]:
				tp_list.append(index - 1)
			else:
				tp_list.append(index - 1)
				tp_list.append(index + 2)

		# todo: 坎张
		# [101]
		if x[3] > 0:
			tp_list.append(index + 1)

	return tp_list

def calc_xts(m, d, if_que):
	"""
	todo: 计算向听数
	"""
	if m + d <= 5:
		c = 0
	else:
		c = m + d - 5
	if m + d <= 4:
		q = 1
	else:
		if if_que:
			q = 1
		else:
			q = 0
	x = 9 - 2 * m - d + c - q
	if d > 1:
		x = x - 1
	return x

def calc_tp_sum(hc: list, tp_list: list):
	"""
	todo: 计算进张所有牌数量
	"""
	my_sum = 0
	for x in tp_list:
		my_sum += 4 - hc[x]
	return my_sum