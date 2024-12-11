# -*- coding: utf-8 -*-

from itertools import groupby

from collections import Counter, defaultdict


class Policy:
	"""
	TODO: 玩家动作选择策略辅助
	"""
	def __init__(self, curr_hand_cards, total_cards):
		"""
		初始化参数
		"""
		self.curr_hand_cards = curr_hand_cards
		self.total_cards = total_cards

	def hand_cards_scores(self):
		"""
		判断手牌分数
		"""
		base_score = [10, 7, 5, 1]
		base_lv = 20
		scores = []
		for i, card1 in enumerate(self.curr_hand_cards):
			score = 0
			for j, card2 in enumerate(self.curr_hand_cards):
				if j != i:
					gap = abs(card1 - card2)
					if gap < 4:
						score += base_score[gap] * base_lv

			for j, card3 in enumerate(self.total_cards):
				if j != i:
					gap = abs(card1 - card3)
					if gap < 4:
						score += base_score[gap]
		return scores

	def calc_total_scores(self):
		"""
		判断一副牌的分数
		"""
		scores = self.hand_cards_scores()
		return sum(scores) / len(scores)

	def calc_is_pong(self, card):
		"""
		根据碰牌前后的分数，判断是否碰牌
		"""
		if self.curr_hand_cards.count(card) < 2:
			return False, -1
		else:
			score0 = self.hand_cards_scores()
			# 碰完，打出一张后的分数的最大值
			self.curr_hand_cards.remove(card)
			self.curr_hand_cards.remove(card)
			max_score = 0
			max_card = -1
			for i in range(self.curr_hand_cards):
				self.curr_hand_cards.remove(i)
				score1 = self.hand_cards_scores()
				if score1 > max_score:
					max_score = score1
					max_card = i
				self.curr_hand_cards.append(i)
		if max_score > score0:
			return True, max_card
		else:
			return False, -1

	def calc_is_gong(self, card):
		"""
		根据杠牌前后的分数判断是否要杠牌
		"""
		if self.curr_hand_cards.count(card) < 3:
			return False
		else:
			score1 = self.hand_cards_scores()
			# 杠完，打出一张牌后的分数的最大值
			self.curr_hand_cards.remove(card)
			self.curr_hand_cards.remove(card)
			self.curr_hand_cards.remove(card)
			max_score = 0
			for i in self.curr_hand_cards:
				self.curr_hand_cards.remove(i)
				score = self.hand_cards_scores()
				if score > max_score:
					max_score = score
				self.curr_hand_cards.append(i)
			if max_score > score1:
				return True
			else:
				return False

def do_remove_cards(curr_hand_cards, card, count=0):
	"""
	删除卡牌
	"""
	for i in range(count):
		curr_hand_cards.remove(card)

def do_add_cards(target_cards, card, count=0):
	"""
	添加卡牌
	"""
	tmp = []
	for i in range(count):
		tmp.append(card)
	target_cards.append(tmp)

def do_add_remove_cards(shun_list, temp_cards, card):
	shun_list.append(card)
	for i in card:
		temp_cards.remove(i)

# TODO: 玩家手牌具体情况
def get_four_same(curr_hand_cards):
	"""
	获取相同的四个数
	"""
	four_cards = []
	counter = Counter(curr_hand_cards)
	for card, nums in counter.items():
		if nums == 4:
			do_add_cards(four_cards, card, 4)
			do_remove_cards(curr_hand_cards, card, 4)
	return four_cards

def get_three_same(curr_hand_cards):
	"""
	获取相同的四个数
	"""
	triple_cards = []
	counter_dict = Counter(curr_hand_cards)
	for card, nums in counter_dict.items():
		if nums == 3:
			do_add_cards(triple_cards, card, 3)
			do_remove_cards(curr_hand_cards, card, 3)
	return triple_cards

def get_pair_same(curr_hand_cards):
	"""
	获取相同的四个数
	"""
	pair_cards = []
	counter = Counter(curr_hand_cards)
	for card, nums in counter.items():
		if nums == 2:
			do_add_cards(pair_cards, card, 2)
			do_remove_cards(curr_hand_cards, card, 2)
	return pair_cards

def calc_gap_cards(tmp_cards):
	""" 计算间隔的两个数 """
	gap_cards = []
	for i in range(len(tmp_cards) - 1):
		if tmp_cards[i] - tmp_cards[i + 1] == -2:
			gap_cards.append(tmp_cards[i])
			gap_cards.append(tmp_cards[i + 1])
		continue
	for card in gap_cards:
		if card in tmp_cards:
			tmp_cards.remove(card)

	return gap_cards

def get_group_0_values(curr_hand_cards):
	"""
	卡牌分组(连续6，连续5，连续4，连续3，连续2, 单)
	"""
	# fun = lambda (i, v): v - i
	# 由于python3中lambda不支持用括号的方式解压，只能这样写
	# fun = lambda x: x[1] - x[0]
	result = []
	for k, g in groupby(enumerate(curr_hand_cards), lambda x: x[1] - x[0]):
		result.append([v for i, v in g])
	return result

def get_group_1_values(tmp_cards):
	""" 获取相邻数 """
	start_index = 0
	result = []
	median = []

	# 索引从start_index起，到最后
	for raw_index in range(len(tmp_cards)):
		# 判断是否for循环到指定位置
		if start_index == raw_index:
			# 初始移动位置参数
			index = 0
			while True:
				# 指针指向的起始值
				start_value = tmp_cards[start_index]
				# 如果指针指向最后一个位置，开始值=最后一个值
				if start_index + index == len(tmp_cards):
					end_value = start_value
				else:
					end_value = tmp_cards[start_index + index]
				# 通过初始值 + 位置参数值 是否等于 最后一个值，判断是否为相邻数，如果是，添加到中间列表
				if start_value + index == end_value:
					median.append(end_value)
					# 位置参数 + 1
					index += 1
				else:
					# 如果不是，初始指针指向 移动位置参数个单位
					start_index += index
					# 把每主相邻数添加到结果列表
					result.append(median)
					median = []
					break
	# 通过高阶函数，对结果集中每个相邻数列表进行插值操作
	return result

def get_two_and_three(tmp_cards):
	"""

	:param tmp_cards:
	:return:
	"""
	continue_two = []
	continue_three = []
	comb_cards = get_group_1_values(tmp_cards)
	for i, cards in enumerate(comb_cards):
		if len(cards) == 2:
			do_add_remove_cards(continue_two, tmp_cards, cards)
		else:
			if len(cards) >= 3:
				do_add_remove_cards(continue_three, tmp_cards, cards[:3])
	return continue_two, continue_three

def calc_two_cards(tmp_cards):
	"""
	"""
	continue_two = []
	comb_cards = get_group_1_values(sorted(tmp_cards))
	for i, cards in enumerate(comb_cards):
		if len(cards) >= 2:
			do_add_remove_cards(continue_two, tmp_cards, cards[:2])

def can_played_cards(curr_hand_cards):
	"""
	计算玩家手牌向听数
	1.不足5block的手牌，向听数=8-2*面子数-搭子数

	2.足够5block的手牌，向听数=3-(顺子/刻子)，没有对子则+1向听

	TODO: 卡牌优先级别
		1.四张
		2.三张
		3.三连续
		4.二连续
		5.对子
		6.间隔
		7.单张
	"""
	legal_cards = []
	# 获取临时卡牌
	tmp_cards = curr_hand_cards.copy()
	# 计算四张
	same_four_count = get_four_same(tmp_cards)
	same_three_count = get_three_same(tmp_cards)
	same_two_count = get_pair_same(tmp_cards)
	two_count, three_count = get_two_and_three(tmp_cards)
	gap_count = calc_gap_cards(tmp_cards)
	legal_cards.extend(same_four_count)
	legal_cards.extend(same_three_count)
	legal_cards.extend(three_count)
	legal_cards.extend(two_count)
	legal_cards.extend(same_two_count)
	legal_cards.append(gap_count)

	if tmp_cards[1:]:
		return tmp_cards[1:]
	elif gap_count:
		return gap_count
	elif two_count:
		return two_count
	elif same_two_count:
		return same_two_count
	elif three_count:
		return three_count
	elif same_three_count:
		return same_three_count
	elif same_four_count:
		return same_four_count

def calc_play_cards(curr_hand_cards, lai_zi):
	"""
	计算玩家胡牌逻辑
	"""
	# 拷贝玩家手牌
	tmp_cards = curr_hand_cards[:]
	cards_dict = {card: tmp_cards.count(card) for card in tmp_cards}
	# 删除癞子牌
	if cards_dict.get(lai_zi, 0):
		tmp_cards.remove(lai_zi)

	# 删除手中的将牌
	calc_que(tmp_cards)

	# 计算剩余的牌，是否能够组成胡牌
	calc_set(tmp_cards)

	# 计算搭子(二连顺，间隔顺)
	calc_two_cards(tmp_cards)
	calc_gap_cards(tmp_cards)

	# 当没有能带出的手牌时，则随机选择一张
	if not tmp_cards:
		return curr_hand_cards

	# 否则，按此部分逻辑出牌
	return tmp_cards

def calc_da_zi(tmp_cards):
	"""
	计算玩家手中搭子数量
	"""
	# 计算二连顺，间隔顺
	calc_two_cards(tmp_cards)
	calc_gap_cards(tmp_cards)

	return tmp_cards

def calc_que(tmp_cards):
	"""
	计算玩家手中的将牌(2张)
	"""
	# 统计每张手牌数量
	# 删除将牌，计算是否能胡
	cards_count = Counter(tmp_cards)
	for card, nums in cards_count.items():
		if nums >= 2:
			for _ in range(2):
				tmp_cards.remove(card)
			break

def calc_set(cards):
	"""
	计算三张卡牌是否为连续值
	"""
	set_count = 0

	# 统计每张卡牌的值
	cards_dict = {card: cards.count(card) for card in cards}
	# 检查手牌中，相同的三张和四张
	for card in cards_dict:
		if cards_dict[card] == 3 or cards_dict[card] == 4:
			set_count += 1
			for _ in range(cards_dict[card]):
				cards.pop(cards.index(card))

	# 计算卡牌类型
	cards_by_type = defaultdict(list)
	for card in cards:
		_type = card // 10
		_value = card % 10
		cards_by_type[_type].append(_value)

	# 根据卡牌类型，计算此类型连续三张卡牌的情况
	for _type in cards_by_type.keys():
		values = sorted(cards_by_type[_type])
		# 当卡牌数量大于2时，说明能够组成连续的牌
		if len(values) > 2:
			for index, _ in enumerate(values):
				# 从第一张卡牌开始计算它后面两张是否为连续卡牌
				# [i, i+1, i+2]
				if index == 0:
					cards_case = [values[index], values[index+1], values[index+2]]
				# 计算完index=0后，开始计算其他剩余的卡牌
				# [i-2, i-1, i]
				elif index == len(values) - 1:
					cards_case = [values[index-2], values[index-1], values[index]]
				# 计算间隔卡牌
				# [i-1, i, i+1]
				else:
					cards_case = [values[index-1], values[index], values[index+1]]

				# 条件满足时，则检查此三张卡牌是否连续
				if check_consecutive(cards_case):
					# 当卡牌连续时，增加1此统计
					set_count += 1
					for card in cards_case:
						values.pop(values.index(card))
						card = (10 * _type) + card
						# 删除手牌中连续的卡牌
						if card in cards:
							cards.pop(cards.index(card))

def check_consecutive(three_list):
	"""
	检查三张卡牌是否为连续卡牌
	"""
	cards = list(map(int, three_list))
	if sorted(cards) == list(range(min(cards), max(cards) + 1)):
		return True
	return False

if __name__ == "__main__":
	# test_cards = [18, 18, 18, 21, 21, 21, 30, 30, 30, 28, 28, 28, 38, 39]
	test_cards = [11, 11, 11, 12, 13, 22, 22, 22, 21, 23, 31, 29, 32, 34]
	# a, b = calc_ting_count(test_cards)
	# print("a: ", a)
	# print("b: ", b)