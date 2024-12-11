# -*- coding: utf-8 -*-

import math

def check_same_area(x, y):
	"""
	todo: 检查两张牌是否在同一区间
	判断两张卡牌是否为同一花色
	"""
	if x < 10 and y < 10:
		return True
	elif 9 <= x < 19 and 9 <= y < 19:
		return True
	elif 18 <= x < 28 and 18 <= y < 28:
		return True
	elif x >= 27 and y >= 27:
		return True

def get_mz_combine(single_color_cards):
	"""
	todo: 获取面子组合卡牌
	找出所有能够组成面子的卡牌
	"""
	single_color_cards = single_color_cards  # [1, 0, 1, 1, 0, 0, 2, 1, 0]
	depth = math.floor(sum(single_color_cards) / 3)
	available = [[0, 0] for _ in range(len(single_color_cards))]
	my_ave_list = []

	# todo: 计算顺子
	# [x][0] 为顺子 -> [1, 2, 3]
	for x in range(len(single_color_cards) - 2):
		if all([single_color_cards[x], single_color_cards[x + 1], single_color_cards[x + 2]]) and \
				math.floor(x / 9) == math.floor((x + 1) / 9) == math.floor((x + 2) / 9):
			available[x][0] = 1

	# todo: 计算刻子
	# [x][1]为刻子 -> [111]
	for x in range(len(single_color_cards)):
		if single_color_cards[x] >= 3:
			available[x][1] = 1

	# todo: 顺子、刻子
	# 添加计算好的顺子和刻子牌，使用元组类型进行存储
	my_ave_list = [(x, available[x][0], 0) for x in range(len(single_color_cards)) if available[x][0] > 0]
	my_ave_list += [(x, 0, available[x][1]) for x in range(len(single_color_cards)) if available[x][1] > 0]
	# my_ave_list = [(2, 1, 0), (3, 1, 0), (4, 1, 0), (4, 0, 1), (5, 0, 1)]

	m = []
	m_list = []
	stack = []
	stack.append(single_color_cards)

	def dfs(m_list, d):
		"""
		dfs搜索, d为depth深度
		"""
		if len(m_list) < depth:
			# 该层被剪掉的分支计数,最大为len(mz_list)
			continue_count = 0
			for x in my_ave_list:
				stack_diff = len(stack) - d - 1  # 需要将栈弹出的次数
				for y in range(stack_diff):
					stack.pop()
				hc = stack[d].copy()  # 当前深度手牌c
				diff = len(m_list) - d  # 如果返回了上一层,自然要弹出在下层添加的面子
				for y in range(diff):
					m_list.pop()

				# 当前组合索引位置
				# index, sz, kz = x[0], x[1], x[2]
				origin_index = x[0]

				# todo: 计算是否存在顺子
				if x[1] > 0:
					# 换位
					hc[origin_index] -= 1
					hc[origin_index + 1] -= 1
					hc[origin_index + 2] -= 1
					if hc[origin_index] >= 0 and hc[origin_index + 1] >= 0 and hc[origin_index + 2] >= 0:
						m_list.append(x)
					else:
						# 复位
						hc[origin_index] += 1
						hc[origin_index + 1] += 1
						hc[origin_index + 2] += 1
						# 分支被剪，计数+1
						continue_count += 1
						# 同一层全被剪 说明路径到头了
						# 连续len(my_ave_list)次continue为终点
						if continue_count >= len(my_ave_list):
							m_list_copy = m_list.copy()
							m_list_copy.sort()
							if m_list_copy not in m:
								m.append(m_list_copy)
						continue

				# todo: 计算是否存在刻子
				elif x[2] > 0:
					# 换位
					hc[origin_index] -= 3
					if hc[origin_index] >= 0:
						m_list.append(x)
					else:
						# 复位
						hc[origin_index] += 3
						# 分支被剪，计数+1
						continue_count += 1
						# 同一层全被剪 说明路径到头了
						# 连续len(my_ave_list)次continue为终点
						if continue_count >= len(my_ave_list):
							m_list_copy = m_list.copy()
							m_list_copy.sort()
							if m_list_copy not in m:
								m.append(m_list_copy)
						continue
				stack.append(hc)
				# todo: dfs搜索， 深度+1
				dfs(m_list, d + 1)
		else:
			# 路径长度达到最大深度
			m_list_copy = m_list.copy()
			m_list_copy.sort()
			if m_list_copy not in m:
				m.append(m_list_copy)

	# todo: dfs搜索
	dfs(m_list, 0)

	# 回退面子牌，应对 -> 1345牌型拆解为: 13, 45 or 1, 345
	# 已经得到了m, 执行面子回退
	mz_count_max = 0
	for x in m:
		mz_count = len(x)
		if mz_count > mz_count_max:
			mz_count_max = mz_count
	for x in m:
		if len(x) == mz_count_max:
			for y in range(len(x)):
				z = x[0: y] + x[y + 1: ]
				z.sort()
				if z not in m:
					m.append(z)
	return m

def get_dz_combine(single_color_cards):
	"""
	todo: 获取搭子组合
	找出所有能够组成搭子的卡牌
	"""
	single_color_cards = single_color_cards
	depth = math.floor(sum(single_color_cards) / 2)
	# x[0], x[1], x[2] = 对子，二连顺，间隔顺
	available = [[0, 0, 0] for _ in range(len(single_color_cards))]
	my_ave_list = []

	# [2]
	# [x][0]为对子
	for x in range(len(single_color_cards)):
		if single_color_cards[x] >= 2:
			available[x][0] = 1

	# [11]
	# [X][1]为二连顺
	for x in range(len(single_color_cards) - 1):
		if x < 27:
			if all([single_color_cards[x], single_color_cards[x + 1]]) and check_same_area(x, x + 1):
				available[x][1] = 1

	# 101
	# [x][2]为间隔顺
	for x in range(len(single_color_cards) - 2):
		if x < 27:
			if all([single_color_cards[x], single_color_cards[x + 2]]) and check_same_area(x, x + 2):
				available[x][2] = 1

	# todo: 对子，二连顺，间隔顺
	my_ave_list = [(x, available[x][0], 0, 0) for x in range(len(single_color_cards)) if available[x][0] > 0]
	my_ave_list += [(x, 0, available[x][1], 0) for x in range(len(single_color_cards)) if available[x][1] > 0]
	my_ave_list += [(x, 0, 0, available[x][2]) for x in range(len(single_color_cards)) if available[x][2] > 0]

	m = []
	stack = []
	m_list = []
	stack.append(single_color_cards)

	def dfs(m_list, d):
		"""
		dfs搜索, d为depth深度
		"""
		if len(m_list) < depth:
			# 该层被剪掉的分支计数,最大为len(mz_list)
			continue_count = 0
			for x in my_ave_list:
				stack_diff = len(stack) - d - 1  # 需要将栈弹出的次数
				for y in range(stack_diff):
					stack.pop()
				hc = stack[d].copy()  # 当前深度手牌c
				diff = len(m_list) - d  # # 如果返回了上一层,自然要弹出在下层添加的面子
				for y in range(diff):
					m_list.pop()

				# 当前组合索引位置
				# index, dz, es, gp = x[0], x[1], x[2], x[3]
				origin_index = x[0]

				# [2]
				# todo: 计算是否存在对子
				if x[1] > 0:
					# 换位
					hc[origin_index] -= 2
					if hc[origin_index] >= 0:
						m_list.append(x)
					else:
						# 复位
						hc[origin_index] += 2
						continue_count += 1
						# 连续len(my_ave_list)次continue为终点
						if continue_count >= len(my_ave_list):
							m_list_copy = m_list.copy()
							m_list_copy.sort()
							if m_list_copy not in m:
								m.append(m_list_copy)
						continue

				# [11]
				# todo: 计算是否存在二连顺
				elif x[2] > 0:
					# 换位
					hc[origin_index] -= 1
					hc[origin_index + 1] -= 1
					if hc[origin_index] >= 0 and hc[origin_index + 1] >= 0:
						m_list.append(x)
					else:
						# 复位
						hc[origin_index] += 1
						hc[origin_index + 1] += 1
						# 分支被剪，计数+1
						continue_count += 1
						# 同一层全被剪 说明路径到头了
						# 连续len(my_ave_list)次continue为终点
						if continue_count >= len(my_ave_list):
							m_list_copy = m_list.copy()
							m_list_copy.sort()
							if m_list_copy not in m:
								m.append(m_list_copy)
						continue

				# todo: 计算是否存在间隔顺
				elif x[3] > 0:
					# 换位
					hc[origin_index] -= 1
					hc[origin_index + 2] -= 1
					if hc[origin_index] >= 0 and hc[origin_index + 2] >= 0:
						m_list.append(x)
					else:
						# 复位
						hc[origin_index] += 1
						hc[origin_index + 2] += 1
						# 分支被剪，计数+1
						continue_count += 1
						# 同一层全被剪 说明路径到头了
						# 连续len(my_ave_list)次continue为终点
						if continue_count >= len(my_ave_list):
							m_list_copy = m_list.copy()
							m_list_copy.sort()
							if m_list_copy not in m:
								m.append(m_list_copy)
						continue
				stack.append(hc)
				# todo: dfs搜索，深度 + 1
				dfs(m_list, d + 1)
		else:
			# 路径长度达到最大深度
			m_list_copy = m_list.copy()
			m_list_copy.sort()
			if m_list_copy not in m:
				m.append(m_list_copy)

	# todo: dfs搜索
	dfs(m_list, 0)

	return m


if __name__ == "__main__":
	hc_list = [1, 0, 1, 1, 1, 0, 2, 1, 0]
	print(get_mz_combine(hc_list))
	print(get_dz_combine(hc_list))