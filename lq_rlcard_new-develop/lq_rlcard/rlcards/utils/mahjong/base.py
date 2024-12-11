# -*- coding: utf-8 -*-

from ...const.mahjong import const
from ...const.mahjong.const import SuitType
from ..mahjong.singleton import Singleton

class RuleBase(metaclass=Singleton):
	"""
	初始化麻将基本规则
	"""
	@staticmethod
	def make_card(suit, value):
		"""
		制作卡牌
		"""
		return suit * 10 + value

	@staticmethod
	def get_value(card):
		"""
		获取卡牌值
		"""
		return (int(card) or 0) % 10

	@staticmethod
	def get_suit(card):
		"""
		获取卡牌位置
		"""
		return (int(card) or 0) // 10

	@staticmethod
	def is_card(card):
		"""
		判断是否为卡牌
		"""
		if not card or not isinstance(card, int):
			return False
		if not 10 <= card <= 80:
			return False
		return card in const.ALL_CARDS

	@staticmethod
	def is_bird(card):
		"""
		判断幺鸡
		"""
		return RuleBase.get_value(card) in const.BIRD_VALUE

	@staticmethod
	def sort(cards):
		"""
		卡牌排序
		"""
		def compare(value):
			if value == const.LAI_ZI:
				return 0
			return value

		cards.sort(key=compare)

	@staticmethod
	def group_by_suit(cards):
		"""
		麻将分组，只有无花色
		"""
		group = dict()
		for card in cards:
			suit = RuleBase.get_suit(card)
			value = RuleBase.get_value(card)
			group.setdefault(suit, []).append(value)

		return group

	@staticmethod
	def is_value_shun_zi(cards):
		"""
		判断牌值是否全部由顺子构成，注意这里不能直接传牌过来，只能传牌值，不能带花色
		所有会改变原参数的值的方法，都应该在开始的时候直接复制list
		"""
		cards = list(cards)
		path = []
		for i in range(0, len(cards) - 1, 3):
			v = cards[0]
			if v in cards and v + 1 in cards and v + 2 in cards:
				cards.remove(v)
				cards.remove(v + 1)
				cards.remove(v + 2)
				path.append([v, v + 1, v + 2])
			else:
				return False, []

		return True, path

	@staticmethod
	def is_value_ke_zi(cards):
		"""
		判断牌值是否为刻子，注意，这里不能直接传牌过去，只能传牌值，不能带花色
		"""
		return len(cards) == 3 and cards[0] == cards[1] == cards[2]

	@staticmethod
	def is_value_gang(cards):
		"""
		判断能不能杠
		"""
		return len(cards) == 4 and cards[0] == cards[1] == cards[2] == cards[3]

	@staticmethod
	def calc_value_count(cards, card):
		"""
		统计、计算给定值在列表中出现的次数
		"""
		count_nums = cards.count(card)
		if count_nums > 0:
			return count_nums
		return 0

	@staticmethod
	def calc_suits(cards):
		"""
		查找指定的牌中所出现的花色列表
		"""
		suits = dict()
		for v in cards:
			suits[RuleBase.get_suit(v)] = 1
		return suits.keys()

	@staticmethod
	def calc_card_suit_count(cards):
		"""
		计算同一种花色出现的次数
		"""
		suits = {1: 0, 2: 0, 3: 0}
		for v in cards:
			suit = RuleBase.get_suit(v)
			if suit in suits:
				suits[suit] += 1
		return suits

	@staticmethod
	def make_wst_cards_by_suits(suits):
		"""
		返回给定花色(万索筒)的所有牌
		"""
		result = list()
		for v in suits:
			if v in SuitType:
				for i in range(1, 10):
					result.append(RuleBase.make_card(v, i))
		return result

	@staticmethod
	def get_card_list_by_count(cards: list, count, with_more=False):
		"""
		统计卡牌列表
		"""
		same_value_list = RuleBase.get_same_value_cards(cards)
		temp_list = []
		for i in same_value_list:
			if len(same_value_list[i]) == count:
				temp_list.append(same_value_list[i])
			if with_more and len(same_value_list[i]) > count:
				temp_list.append(same_value_list)
		return temp_list

	@staticmethod
	def get_same_value_cards(cards: list):
		"""
		获取相同的卡牌值
		"""
		same_list = {}
		for card in cards:
			same_list.setdefault(card, []).append(card)
		return same_list

	@staticmethod
	def remove_by_value(card_data, value, remove_count=1):
		"""
		remove_count为-1时，表示删除全部，默认值为1
		"""
		if card_data is None:
			return 0
		if isinstance(card_data, int):
			card_data = [card_data]
		data_len = len(card_data)
		count = remove_count == -1 and data_len or remove_count
		already_remove_count = 0
		for i in range(0, count):
			if value in card_data:
				already_remove_count += 1
				if already_remove_count >= remove_count:
					break
			else:
				break
		return already_remove_count