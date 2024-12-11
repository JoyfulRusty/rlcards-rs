# -*- coding: utf-8 -*-


from copy import deepcopy


from rlcards.games.sytx_gz import config
from rlcards.const.sytx_gz.const import *
from rlcards.utils.mahjong.singleton import Singleton


class Rules(metaclass=Singleton):
	"""
	水鱼规则
	"""
	# 普通扑克
	__pokers = [
		101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,  # 方块
		201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213,  # 梅花
		301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313,  # 红心
		401, 402, 403, 404, 405, 406, 407, 408, 409, 411, 412, 413,  # 黑桃
	]
	# 万能牌(癞子)
	__lai_zi = (515, 516, 517)  # 元组不可变

	@staticmethod
	def make_pokers(with_lai_zi=0):
		"""
		制作扑克
		"""
		cards = []
		cards.extend(Rules.__pokers)
		if with_lai_zi == LzType.MH:
			cards.extend(Rules.__lai_zi[:2])
		elif with_lai_zi == LzType.GX:
			cards.extend(Rules.__lai_zi)
		return cards

	@staticmethod
	def make_card(s, v):
		"""
		制作卡牌
		"""
		return s * 100 + v

	@staticmethod
	def lz_cards():
		"""
		癞子卡牌
		"""
		return list(Rules.__lai_zi)

	@staticmethod
	def is_cards(card):
		"""
		判断是否为卡牌
		"""
		if not card or not isinstance(card, int):
			return False
		if not 101 <= card <= 413:
			return False
		return card in Rules.__pokers

	@staticmethod
	def suit(c):
		"""
		卡牌花色
		"""
		return int(c / 100)

	@staticmethod
	def value(c):
		"""
		卡牌值
		"""
		return c % 100

	@staticmethod
	def value_and_suit(c):
		return c % 100, c // 100

	@staticmethod
	def is_poker(c):
		"""
		判断是否为扑克牌
		"""
		return c in Rules.__pokers or (c % 100 != 14 and 1 <= c % 100 <= 16)

	@staticmethod
	def abstract_values(cards: list):
		"""
		提取卡牌值
		"""
		return list(map(Rules.value, cards))

	@staticmethod
	def abstract_suits(cards: list):
		"""
		提取卡牌位置
		"""
		result = []
		for card in cards:
			result.append(Rules.suit(card))
		return result

	@staticmethod
	def abstract_value_to_dan_zhang(cards):
		"""
		提取值为单张的卡牌
		"""
		result = []
		for card in cards:
			value = Rules.value(card)
			if value > 10:
				value = value % 10 / 10
			result.append(value)
		return result

	@staticmethod
	def __abstract_value_to_dz(card):
		"""
		单张卡牌值
		"""
		value = card % 100
		if value > 10:
			value = value % 10 / 10
		return value

	@staticmethod
	def make_cards_by_suits(suits):
		"""
		返回给定花色的所有卡牌
		"""
		result = list()
		suit = list(CARD_TYPES.keys())
		for s in suits:
			if s in suit:
				for v in range(14):
					if v == 10:
						continue
					result.append(Rules.make_card(s, v))
		return result

	@staticmethod
	def calc_dan_zhang_value(card):
		"""
		计算单张卡牌值
		"""
		return sum(Rules.abstract_value_to_dan_zhang(card)) % 10

	@staticmethod
	def match_dan_zhang(card, result):
		"""
		匹配单张卡牌值
		"""
		result[0] = SyType.SINGLE
		result[1] = [Rules.calc_dan_zhang_value(card)]

	@staticmethod
	def check_has_lai_zi(cards):
		"""
		检查是否存在癞子
		"""
		lz_cards = Rules.lz_cards()
		lai_zi_cards = []
		for c in cards:
			if c in lz_cards:
				lai_zi_cards.append(c)
		return lai_zi_cards

	@staticmethod
	def check_sy_or_tx(cards):
		"""
		检查水鱼或水鱼天下
		"""
		result = [0, []]
		res, cards_v_list = Rules.get_count_list_by_value(cards)
		res_len = len(res)
		if res_len == 1:
			# 水鱼天下
			result[0] = SyType.SY_TX
			result[1] = Rules.get_aa_value(res, cards_v_list[0: 1])
		elif res_len == 2 and max(res.values()) == 2:
			cards_v_list.sort()
			# 水鱼
			result[0] = SyType.SY
			result[1] = Rules.get_aa_value(res, cards_v_list[0: 1])
			result[1].append(cards_v_list[-1])
			result[1].sort()
		return result

	@staticmethod
	def check_sy_tx_after_sp_by_lai_zi(cards):
		"""
		检查水鱼天下后癞子
		"""
		# 检查是否存在癞子
		p_lz = Rules.check_has_lai_zi(cards)
		if not p_lz:
			# 检查水鱼或水鱼天下
			return Rules.check_sy_or_tx(cards)
		cards_copy = deepcopy(cards)
		for n in p_lz:
			cards_copy.remove(n)
		res, cards_v_list = Rules.get_count_list_by_value(cards_copy)
		res_len = len(res)
		result = [0, []]

		# 水鱼天下清空单独处理(包含三张癞子情况)
		if res_len == 1:
			result[0] = SyType.SY_TX
			result[1] = Rules.get_aa_value(res, cards_v_list[0: 1])
			return result
		lz_len = len(p_lz)
		if lz_len == 1:
			if res_len == 2 and max(res.values()) == 2:
				pair_cards = [k for k, v in res.items() if v == 2]
				single_cards = [k for k, v in res.items() if v == 1]
				result[0] = SyType.SY
				result[1] = Rules.get_aa_value(res, pair_cards)
				result[1].extend(Rules.get_aa_value(res, single_cards))
				result[1].sort()
			return result

		# 该为撒扑前就判断是否组成水鱼或水鱼天下
		elif lz_len == 2:
			cards_v_list.sort()
			result[0] = SyType.SY
			result[1] = Rules.get_aa_value(res, cards_v_list[0: 1])
			result.append(cards_v_list[-1])
			result[1].sort()
		return result

	@staticmethod
	def is_sy_or_sy_tx(cards):
		"""
		癞子版本检查水鱼或水鱼天下，只需要传入癞子即可
		"""
		return Rules.check_sy_tx_after_sp_by_lai_zi(cards)

	@staticmethod
	def get_count_list_by_value(cards):
		"""
		统计列表中的值
		"""
		result = {}
		cards_value_list = Rules.abstract_values(cards)
		for c in cards_value_list:
			result[c] = result.get(c, 0) + 1
		return result, cards_value_list

	@staticmethod
	def get_aa_value(card_dict, cards_v_list, a_val=1):
		"""
		值
		"""
		if not card_dict.get(a_val) or a_val not in cards_v_list:
			result = cards_v_list
		else:
			result = [14]
		return result

	@staticmethod
	def get_type_from_finished_pei_pai(cards: list):
		"""
		从配牌中获取类型
		"""
		# 检查癞子
		card_lz = Rules.check_has_lai_zi(cards)
		if card_lz:
			res_card = list(set(cards).difference(card_lz))
			if not res_card:
				return SyType.PAIR, [16]
			res_val = Rules.abstract_values(res_card)
			return SyType.PAIR, res_val if res_val[0] != 1 else [14]
		result = [0, []]
		res, cards_v_list = Rules.get_count_list_by_value(cards)
		if len(res) == 1:
			result[0] = SyType.PAIR
			result[1] = Rules.get_aa_value(res, cards_v_list[0: 1])
		else:
			# 匹配单张卡牌
			Rules.match_dan_zhang(cards, result)
		return result

	@staticmethod
	def is_single(cards):
		"""
		判断单张
		"""
		cards_type, _ = Rules.get_type_from_finished_pei_pai(deepcopy(cards))
		return cards_type == SyType.SINGLE

	@staticmethod
	def is_pair(cards):
		"""
		判断对子
		"""
		cards_type, _ = Rules.get_type_from_finished_pei_pai(deepcopy(cards))
		return cards_type == SyType.PAIR

	@staticmethod
	def is_sy(cards):
		"""
		判断水鱼
		"""
		cards_type, _ = Rules.is_sy_or_sy_tx(deepcopy(cards))
		return cards_type == SyType.SY

	@staticmethod
	def is_sy_tx(cards):
		"""
		判断水鱼天下
		"""
		cards_type, _ = Rules.is_sy_or_sy_tx(deepcopy(cards))
		return cards_type == SyType.SY_TX

	@staticmethod
	def calc_is_sy_or_sy_tx(cards):
		"""
		计算满足水鱼或水鱼天下返回True
		"""
		cards_type, _ = Rules.is_sy_or_sy_tx(deepcopy(cards))
		return cards_type == SyType.SY_TX or cards_type == SyType.SY

	@staticmethod
	def is_biggest_dan_zhang(cards, card):
		"""
		计算最大的单张
		"""
		card_val = card[0] % 100
		for card in cards:
			if card % 100 > card_val:
				return False
		return True

	@staticmethod
	def is_all_pokers(cards):
		"""
		判断是否都是扑克
		"""
		if not cards or type(cards) is not list:
			return False
		for card in cards:
			if not Rules.is_poker(card):
				return False
		return True

	@staticmethod
	def compare_one_card(cards):
		"""
		比较两张卡牌，大在前，小在后
		"""
		c_lst = Rules.abstract_values(cards)
		if c_lst[0] != c_lst[-1]:
			if c_lst[0] > c_lst[-1]:
				return cards
			else:
				cards[0], cards[-1] = cards[-1], cards[0]
				return cards
		else:
			s_lst = Rules.abstract_suits(cards)
			if s_lst[0] > s_lst[1]:
				return cards
			else:
				cards[0], cards[-1] = cards[-1], cards[0]
				return cards

	@staticmethod
	def __compare_pair(data1, data2, card1, card2):
		"""
		比较对子
		"""
		if data1[0] > data2[0]:
			return IS_MORE
		elif data1[0] == data2[0]:
			return IS_MORE if card1[1] > card2[1] else IS_LESS
		else:
			return IS_LESS

	@staticmethod
	def compare_big_or_small_suit(card1, card2, is_big=False):
		"""
		比较更大或更小
		"""
		v1 = Rules.abstract_values(card1)
		v2 = Rules.abstract_values(card2)
		s1 = Rules.abstract_suits(card1)
		s2 = Rules.abstract_suits(card2)
		if is_big:
			suit_index1 = v1.index(max(v1))
			suit_index2 = v2.index(max(v2))
		else:
			suit_index1 = v1.index(min(v1))
			suit_index2 = v2.index(min(v2))
		return s1[suit_index1] > s2[suit_index2]

	@staticmethod
	def __compare_suit(card1, card2, is_big=False):
		"""
		比较位置
		"""
		if Rules.compare_big_or_small_suit(card1, card2, is_big):
			return IS_MORE
		else:
			return IS_LESS

	@staticmethod
	def __compare_single(data1, data2, card1, card2):
		"""
		比较单张
		"""
		# 当两扑牌点数相同时，比较小扑
		card1_value = Rules.abstract_values(card1)
		card2_value = Rules.abstract_values(card2)
		if data1[0] == data2[0]:
			if type(data1[0]) is int:
				if 10 in card1_value and min(card2_value) not in (1, 9):
					return IS_MORE
				elif min(card1_value) not in (1, 9) and 10 in card2_value:
					return IS_LESS
				elif min(card1_value) < min(card2_value):  # 比小张
					return IS_MORE
				elif min(card1_value) == min(card2_value):  # 比小张花色
					if min(card1_value) == 1:
						return Rules.__compare_suit(card1, card2, is_big=False)
					return Rules.__compare_suit(card1, card2, is_big=config.is_bigger)
				else:
					return IS_LESS
			if type(data1[0]) is float:
				if min(card1_value) == 1:
					return Rules.__compare_suit(card1, card2, is_big=False)
				return Rules.__compare_suit(card1, card2, is_big=config.is_bigger)
		elif int(data1[0]) == int(data2[0]) and (min(card1_value) == 1 or min(card2_value) == 1):
			if min(card1_value) == min(card2_value):  # a1 cQ <==> b1 cK
				return Rules.__compare_suit(card1, card2, is_big=False)
			return IS_MORE if min(card1_value) == 1 else IS_LESS
		elif data1[0] > data2[0]:
			return IS_MORE
		else:
			return IS_LESS

	@staticmethod
	def __compare_te_shu(type1, type2, data1, data2, card1=None, card2=None):
		"""
		比较两组都是特殊牌型大小
		"""
		if type1 > type2:
			return IS_MORE
		if type1 < type2:
			return IS_LESS
		else:
			if type1 == SyType.SY_TX:
				if data1[0] > data2[0]:
					return IS_MORE
				else:
					return IS_LESS
			else:
				# 小扑与大扑值相同比较花色
				if data1 == data2:
					x_b_max_suit = max(Rules.abstract_suits(card1[0:2]))  # 大扑大的花色
					z_b_max_suit = max(Rules.abstract_suits(card2[0:2]))  # 小扑小的花色
					x_s_max_suit = max(Rules.abstract_suits(card1[2:]))
					z_s_max_suit = max(Rules.abstract_suits(card2[2:]))
					if x_b_max_suit > z_b_max_suit and x_s_max_suit > z_s_max_suit:
						return IS_MORE
					elif x_b_max_suit > z_b_max_suit and x_s_max_suit < z_s_max_suit or x_b_max_suit < z_b_max_suit and \
							x_s_max_suit > z_s_max_suit:
						return IS_DRAW
					else:
						return IS_LESS
				# 闲庄各大一扑
				elif data1[0] > data2[0] and data1[1] < data2[1] or data1[0] < data2[0] and data1[1] > data2[1]:
					return IS_DRAW
				# 闲小扑比庄家大，大扑值与庄相同
				elif data1[0] > data2[0] and data1[1] == data2[1]:
					x_b_max_suit = max(Rules.abstract_suits(card1[0:2]))  # 大扑大的花色
					z_b_max_suit = max(Rules.abstract_suits(card2[0:2]))  # 小扑小的花色
					if x_b_max_suit > z_b_max_suit:  # 比较大扑花色
						return IS_MORE
					else:
						return IS_DRAW
				# 闲小扑比庄小，大扑值与庄相同
				elif data1[0] < data2[0] and data1[1] == data2[1]:
					x_b_max_suit = max(Rules.abstract_suits(card1[0:2]))  # 大扑大的花色
					z_b_max_suit = max(Rules.abstract_suits(card2[0:2]))  # 小扑小的花色
					if x_b_max_suit > z_b_max_suit:  # 比较大扑花色
						return IS_DRAW
					else:
						return IS_LESS
				# 闲小扑值与庄相同，大扑比庄大
				elif data1[0] == data2[0] and data1[1] > data2[1]:
					x_s_max_suit = max(Rules.abstract_suits(card1[2:]))
					z_s_max_suit = max(Rules.abstract_suits(card2[2:]))
					if x_s_max_suit > z_s_max_suit:  # 比较小扑花色
						return IS_MORE
					else:
						return IS_DRAW
				# 闲小扑与庄相同，大扑比庄大
				elif data1[0] == data2[0] and data1[1] < data2[1]:
					x_s_max_suit = max(Rules.abstract_suits(card1[2:]))
					z_s_max_suit = max(Rules.abstract_suits(card2[2:]))
					if x_s_max_suit > z_s_max_suit:  # 比较小扑花色
						return IS_DRAW
					else:
						return IS_LESS
				# 闲大扑与小扑都比庄大
				elif data1[0] > data2[0] and data1[1] > data2[1]:
					return IS_MORE
				else:
					return IS_LESS

	@staticmethod
	def __compare_by_data(type1, data1, type2, data2, card1, cards2):
		"""
		实际比较算法
		普通牌型比较，需要分开比较，大比大，小比小
		"""
		is_a_pair1 = type1 == SyType.PAIR
		is_a_pair2 = type2 == SyType.PAIR
		if is_a_pair1 and not is_a_pair2:
			return IS_MORE
		elif not is_a_pair1 and is_a_pair2:
			return IS_LESS
		elif is_a_pair1 and is_a_pair2:
			return Rules.__compare_pair(data1, data2, card1, cards2)  # 比较对子
		else:
			return Rules.__compare_single(data1, data2, card1, cards2)  # 比较单张

	@staticmethod
	def compare_by_te_shu(type1, type2, data1, data2):
		"""
		特殊牌型比较(水鱼或水鱼天下)
		"""
		is_ts_sy1 = type1 in [SyType.SY, SyType.SY_TX]
		is_ts_sy2 = type2 in [SyType.SY, SyType.SY_TX]
		if is_ts_sy1 and not is_ts_sy2:  # 组1为特殊牌型，组2不为特殊牌型
			return IS_MORE
		if not is_ts_sy1 and is_ts_sy2:  # 组2为特殊牌型，组1不为特殊牌型
			return IS_LESS
		if is_ts_sy1 and is_ts_sy2:  # 两组都是特殊牌型
			return Rules.__compare_te_shu(type1, type2, data1, data2)

	@staticmethod
	def compare_one_combine(cards1, cards2):
		"""
		比较组合
		cards1: 庄家牌
		cards2: 闲家牌
		is_pei_pai: 配过牌的基础上比较
		"""
		if not cards1 or not cards2:
			raise ValueError("cards is an empty sequence: {}, {}".format(cards1, cards2))
		cards1.sort()
		cards2.sort()
		type1, data1 = Rules.get_type_from_finished_pei_pai(cards1)
		type2, data2 = Rules.get_type_from_finished_pei_pai(cards2)
		return Rules.__compare_by_data(type1, data1, type2, data2, cards1, cards2)

	@staticmethod
	def compare_one_by_lai_zi(cards1, cards2, with_lai_zi=None):
		"""
		cards1: 闲家大扑/小扑
		cards2: 庄家大扑/小扑
		"""
		if not cards1 or not cards2:
			raise ValueError("cards is an empty sequence")
		if not with_lai_zi:
			with_lai_zi = Rules.lz_cards()
		cards1.sort()
		cards2.sort()
		card1_lz = Rules.check_has_lai_zi(cards1)
		card2_lz = Rules.check_has_lai_zi(cards2)
		if not card1_lz and not card2_lz:
			type1, data1 = Rules.get_type_from_finished_pei_pai(cards1)
			type2, data2 = Rules.get_type_from_finished_pei_pai(cards2)
			return Rules.__compare_by_data(type1, data1, type2, data2, cards1, cards2)
		if len(card1_lz) == len(with_lai_zi):  # 两张癞子的豹最大
			return IS_MORE
		if len(card2_lz) == len(with_lai_zi):
			return IS_LESS
		type1, data1 = Rules.get_type_from_finished_pei_pai(cards1)
		type2, data2 = Rules.get_type_from_finished_pei_pai(cards2)

		is_a_pair1 = type1 == SyType.PAIR
		is_a_pair2 = type2 == SyType.PAIR
		if is_a_pair1 and not is_a_pair2:
			return IS_MORE
		elif not is_a_pair1 and is_a_pair2:
			return IS_LESS
		elif is_a_pair1 and is_a_pair2:
			if data1[0] > data2[0]:
				return IS_MORE
			elif data1[0] == data2[0]:
				if card1_lz and card2_lz:  # todo: 待定
					return IS_MORE if cards1[0] > cards2[0] else IS_LESS
				if not card1_lz:
					return IS_MORE
				if not card2_lz:
					return IS_LESS
				return IS_MORE if cards1[0] > cards2[0] else IS_LESS
			else:
				return IS_LESS

	@staticmethod
	def compare_two_combine(cards1_combined, cards2_combined, with_lai_zi=False):
		"""
		比较两张组合
		"""
		bigger1, bigger2 = cards1_combined[:2], cards2_combined[:2]
		small1, small2 = cards1_combined[2:], cards2_combined[2:]
		if with_lai_zi:
			with_lai_zi = Rules.lz_cards()
			if Rules.check_has_lai_zi(bigger1) or Rules.check_has_lai_zi(bigger2):
				compare_big = Rules.compare_one_by_lai_zi(bigger1, bigger2, with_lai_zi)
			else:
				compare_big = Rules.compare_one_combine(bigger1, bigger2)
		else:
			compare_big = Rules.compare_one_combine(bigger1, bigger2)
		compare_small = Rules.compare_one_combine(small1, small2)
		return compare_big, compare_small

	@staticmethod
	def compare_two_combine_by_lai_zi(cards1_combined, cards2_combined, with_lai_zi):
		"""
		比较两张包含癞子组合
		"""
		bigger1, bigger2 = cards1_combined[:2], cards2_combined[:2]
		small1, small2 = cards1_combined[2:], cards2_combined[2:]
		compare_big = Rules.compare_one_by_lai_zi(bigger1, bigger2, with_lai_zi)
		compare_small = Rules.compare_one_by_lai_zi(small1, small2, with_lai_zi)
		return compare_big, compare_small

	@staticmethod
	def search_combine_index(card_combine):
		"""
		搜索组合索引
		"""
		if Rules.compare_one_combine(card_combine[0:2], card_combine[2:]) == IS_MORE:
			return 1
		else:
			return -1

	@staticmethod
	def search_bigger_small_combine(card_combine):
		"""
		搜索较大组合
		"""
		res = Rules.search_combine_index(card_combine)
		if res == 1:
			bigger = card_combine[:2]
			small = card_combine[2:]
		else:
			bigger = card_combine[2:]
			small = card_combine[:2]
		return bigger, small

	@staticmethod
	def need_compare_two(player_card, deal_card):
		"""
		闲比庄比2铺时，庄只要赢一铺即赢
		"""
		result = Rules.compare_two_combine(player_card, deal_card)
		return DEAL_WIN if IS_LESS in result else PLAYER_WIN

	@staticmethod
	def judge_winner_or_loser(player_card, deal_card, compare_num):
		"""
		赢家或输家
		"""
		if compare_num == COMPARE_Z_SHA_X_FAN_NUM:  # 庄杀闲反
			result = Rules.compare_two_combine(player_card, deal_card)
			return PLAYER_WIN if IS_MORE in result else DEAL_WIN
		elif compare_num == COMPARE_Z_ZOU_X_FAN_NUM:  # 庄走闲反
			return Rules.need_compare_two(player_card, deal_card)
		elif compare_num == COMPARE_X_MI_Z_KAI_NUM:  # 闲密庄开
			return Rules.need_compare_two(player_card, deal_card)
		elif compare_num == COMPARE_X_QG_Z_KAI_NUM:  # 强攻
			return Rules.need_compare_two(player_card, deal_card)
		elif compare_num == COMPARE_Z_SHA_X_XIN_NUM:  # 庄杀闲认
			return DEAL_WIN
		elif compare_num == COMPARE_X_MI_Z_XIN_NUM:  # 闲密庄认
			return PLAYER_WIN
		else:
			return DEAL_WIN  # 闷赔

	@staticmethod
	def judge_winner_or_loser_by_lz(player_card, deal_card, compare_num, with_lai_zi: list):
		"""
		包含癞子时，赢家或输家
		"""
		if compare_num == COMPARE_Z_SHA_X_XIN_NUM:  # 庄杀闲认
			return DEAL_WIN
		elif compare_num == COMPARE_X_MI_Z_XIN_NUM:  # 闲密庄认
			return PLAYER_WIN
		res = Rules.compare_two_combine_by_lai_zi(player_card, deal_card, with_lai_zi)
		if compare_num == COMPARE_Z_ZOU_X_FAN_NUM:  # 庄走闲反
			return DEAL_WIN if IS_LESS in res else PLAYER_WIN
		elif compare_num == COMPARE_Z_SHA_X_FAN_NUM:  # 庄杀闲反
			return PLAYER_WIN if IS_MORE in res else DEAL_WIN
		elif compare_num == COMPARE_X_MI_Z_KAI_NUM:  # 闲密庄开
			return DEAL_WIN if IS_LESS in res else PLAYER_WIN
		elif compare_num == COMPARE_X_QG_Z_KAI_NUM:  # 强攻
			return DEAL_WIN if IS_LESS in res else PLAYER_WIN
		else:
			return DEAL_WIN  # 闷赔

	@staticmethod
	def find_suit_by_val(cards, with_lai_zi):
		"""
		在有癞子牌情况下，找出大小铺中花色最大的
		cards: [大铺 小铺] <-> [癞子 x1, x2,x2] <-> [x1, x1, 癞子, x2] <-> [癞子1 x1, 癞子2 x2]
		"""
		res_cards = list(set(cards).difference(with_lai_zi))
		res_cards_len = len(res_cards)
		if res_cards_len == len(cards):
			b_max_suit = max(Rules.abstract_suits(cards[0:2]))  # 大铺大的花色
			s_max_suit = max(Rules.abstract_suits(cards[2:]))
		else:
			if res_cards_len == 3:
				if cards[0] in with_lai_zi:
					b_max_suit = Rules.suit(cards[1])
					s_max_suit = max(Rules.abstract_suits(cards[2:]))
				else:
					b_max_suit = max(Rules.abstract_suits(cards[:2]))
					s_max_suit = Rules.suit(cards[-1])
			else:
				b_max_suit = Rules.suit(res_cards[0])  # 大铺大的花色
				s_max_suit = Rules.suit(res_cards[1])
		return b_max_suit, s_max_suit

	@staticmethod
	def __compare_te_shu_by_lai_zi(type1, type2, data1, data2, cards1, cards2, with_lai_zi):
		"""
		癞子牌情况下比较特殊牌型
		"""
		if type1 > type2:
			return IS_MORE
		elif type1 < type2:
			return IS_LESS
		if type1 == SyType.SY_TX:
			if data1[0] > data2[0]:
				return IS_MORE
			return IS_LESS
		# 水鱼
		# 1.闲庄各大一铺
		try:
			if data1[0] > data2[0] and data1[1] < data2[1] or data1[0] < data2[0] and data1[1] > data2[1]:
				return IS_DRAW
		except IndexError:
			print(cards1, cards2)
		x_b_max_suit, x_s_max_suit = Rules.find_suit_by_val(cards1, with_lai_zi)  # 闲大铺花色 小铺花色
		z_b_max_suit, z_s_max_suit = Rules.find_suit_by_val(cards2, with_lai_zi)
		# 2.值相同，比较花色
		if data1 == data2:
			if x_b_max_suit > z_b_max_suit and x_s_max_suit > z_s_max_suit:
				return IS_MORE
			elif x_b_max_suit > z_b_max_suit and x_s_max_suit < z_s_max_suit or x_b_max_suit < z_b_max_suit and \
					x_s_max_suit > z_s_max_suit:
				return IS_DRAW
			else:
				return IS_LESS
		# 3.闲小铺比庄大, 大铺值与庄相同
		elif data1[0] > data2[0] and data1[1] == data2[1]:
			if x_b_max_suit > z_b_max_suit:  # 比较大铺花色
				return IS_MORE
			else:
				return IS_DRAW
		# 4.闲小铺比庄小, 大铺值与庄相同
		elif data1[0] < data2[0] and data1[1] == data2[1]:
			if x_b_max_suit > z_b_max_suit:  # 比较大铺花色
				return IS_DRAW
			else:
				return IS_LESS
		# 5.闲小铺值与庄相同, 大铺比庄大
		elif data1[0] == data2[0] and data1[1] > data2[1]:
			if x_s_max_suit > z_s_max_suit:  # 比较小铺花色
				return IS_MORE
			else:
				return IS_DRAW
		# 6.闲小铺值与庄相同, 大铺比庄小
		elif data1[0] == data2[0] and data1[1] < data2[1]:
			if x_s_max_suit > z_s_max_suit:  # 比较小铺花色
				return IS_DRAW
			else:
				return IS_LESS
		# 7.闲大铺与小铺都比庄大
		elif data1[0] > data2[0] and data1[1] > data2[1]:
			return IS_MORE
		else:
			return IS_LESS

	@staticmethod
	def compare_shui_yu(card1, card2):
		type1, data1 = Rules.is_sy_or_sy_tx(card1)
		type2, data2 = Rules.is_sy_or_sy_tx(card2)
		return Rules.__compare_te_shu(type1, type2, data1, data2, card1, card2)

	@staticmethod
	def compare_sy_by_lai_zi(cards1, cards2, with_lai_zi):
		type1, data1 = Rules.is_sy_or_sy_tx(cards1)
		type2, data2 = Rules.is_sy_or_sy_tx(cards2)
		return Rules.__compare_te_shu_by_lai_zi(type1, type2, data1, data2, cards1, cards2, with_lai_zi)

	@staticmethod
	def is_bigger_old(player_card, deal_card, compare_num=-1):
		"""
		癞子版比较只需传with_lai_zi参数即可，经典版本无需传
		with_lai_zi: 癞子牌 <-> [515, 516], poker类中有定义
		"""
		x_cards = deepcopy(player_card)
		z_cards = deepcopy(deal_card)
		if compare_num == COMPARE_Z_ZOU_X_XIN_NUM:  # 庄走闲信
			return DRAW
		elif compare_num == COMPARE_X_SY:  # 闲水鱼
			return True
		elif Rules.check_has_lai_zi(x_cards) or Rules.check_has_lai_zi(z_cards):
			return Rules.is_bigger_by_lai_zi(x_cards, z_cards, compare_num)
		elif compare_num == COMPARE_Z_AND_X_SY:  # 庄闲水鱼
			result = Rules.compare_shui_yu(x_cards, z_cards)
			if result == IS_DRAW:
				return DRAW
			return True if result == IS_MORE else False
		result = Rules.judge_winner_or_loser(x_cards, z_cards, compare_num)  # 普通牌型比较
		return True if result == PLAYER_WIN else False

	@staticmethod
	def is_bigger(player_card, deal_card, compare_num=-1, is_gx=False):
		"""
		癞子版比较只需传with_lai_zi参数即可，经典版本无需传
		with_lai_zi: 癞子牌 <-> [515, 516], poker类中有定义
		is_gx: 是否是广西玩法
		"""
		x_cards = deepcopy(player_card)
		z_cards = deepcopy(deal_card)
		if compare_num == COMPARE_Z_ZOU_X_XIN_NUM:  # 庄走闲信
			return DRAW
		elif compare_num == COMPARE_X_SY:  # 闲水鱼
			if is_gx:
				if Rules.is_chou_10(deal_card):
					return False
			return True
		elif compare_num == COMPARE_Z_SY:
			if is_gx:
				if Rules.is_chou_10(player_card):
					return True
			return False
		elif Rules.check_has_lai_zi(x_cards) or Rules.check_has_lai_zi(z_cards):
			return Rules.is_bigger_by_lai_zi(x_cards, z_cards, compare_num)
		elif compare_num == COMPARE_Z_AND_X_SY:  # 庄闲水鱼
			result = Rules.compare_shui_yu(x_cards, z_cards)
			if result == IS_DRAW:
				return DRAW
			return True if result == IS_MORE else False
		result = Rules.judge_winner_or_loser(x_cards, z_cards, compare_num)  # 普通牌型比较
		return True if result == PLAYER_WIN else False

	@staticmethod
	def is_chou_10(cards):
		""" 是否是臭十 """
		card1_val = Rules.value(cards[0])
		card2_val = Rules.value(cards[1])
		if card1_val == card2_val:
			return False
		if card1_val + card2_val != 10:
			return False
		card3_val = Rules.value(cards[2])
		card4_val = Rules.value(cards[3])
		if card3_val + card4_val != 10:
			return False
		return True

	@staticmethod
	def is_bigger_by_lai_zi(player_card, deal_card, compare_num=-1):
		"""
		有癞子牌存在比大小
		player_card / deal_card必须有癞子牌
		该比较是闲家相对于庄家来比较的
		"""
		with_lai_zi = Rules.lz_cards()
		if compare_num == COMPARE_Z_AND_X_SY:
			result = Rules.compare_sy_by_lai_zi(player_card, deal_card, with_lai_zi)
			if result == IS_DRAW:
				return DRAW
			return True if result == IS_MORE else False
		result = Rules.judge_winner_or_loser_by_lz(player_card, deal_card, compare_num, with_lai_zi)
		return True if result == PLAYER_WIN else False

	@staticmethod
	def literal_big_card(sa_pu):
		"""
		计算较大卡牌类型
		"""
		c_type, _ = Rules.is_sy_or_sy_tx(deepcopy(sa_pu))
		card_type = ""
		if not c_type:
			card_values = Rules.abstract_values(sa_pu[:2])
			card_values.sort(reverse=True)
			if sum(card_values) == 9 and card_type[1] == 1:
				card_type = "幺九八"
			elif card_values[0] == card_values[1]:
				card_type = "对子"
		else:
			if c_type == SyType.SY:
				card_type = "水鱼"
			else:
				card_type = "水鱼天下"
		return card_type

	@staticmethod
	def detect_cc(sa_pu: list):
		"""
		检测臭臭扑
		"""
		if not sa_pu or len(sa_pu) != 4:
			return False
		big_value = Rules.abstract_value_to_dan_zhang(sa_pu[:2])
		small_value = Rules.abstract_value_to_dan_zhang(sa_pu[2:])
		big_value.sort(reverse=True)
		if big_value[0] == 1 and big_value[1] < 1 and small_value[0] < 1 and small_value[1] < 1:
			return True
		return False

	@staticmethod
	def detect_ybg(cards, sa_pu_op, le_num=9):
		"""
		检测阴包谷
		"""
		if sa_pu_op != SA_PU_MI:
			return False
		if not cards or len(cards) != 4:
			return False
		# 计算大牌
		big_value = Rules.abstract_value_to_dan_zhang(cards[:2])
		if big_value[0] == big_value[1]:
			return False
		if sum(big_value) < le_num:
			return True
		return False

	@staticmethod
	def detect_d_yuan_hou(cards, sa_pu_op, le_num=5):
		"""
		检测定远侯，大扑小于5点时，摸牌被认
		"""
		if sa_pu_op != SA_PU_MI:
			return False
		if not cards or len(cards) != 4:
			return False
		small_value = Rules.abstract_value_to_dan_zhang(cards[2:])
		if small_value[0] == small_value[1]:
			return False
		if sum(small_value) < le_num:
			return True
		return False

	@staticmethod
	def detect_h_yin_hou(cards, sa_pu_op, le_num=4):
		"""
		检测淮阴侯，大小扑大于4点让走
		"""
		if sa_pu_op != SA_PU_LP:
			return False
		if not cards or len(cards) != 4:
			return False
		small_value = Rules.abstract_value_to_dan_zhang(cards[2:])
		if small_value[0] == small_value[1]:
			return False
		if sum(small_value) < le_num:
			return True
		return False

	@staticmethod
	def sort_big_small_by_lai_zi(card, sa_pu_op=SA_PU_LP):
		""" 玩家撒铺后才进行排序 """
		if not card:
			return
		if sa_pu_op == SA_PU_SY:
			p_lz = Rules.check_has_lai_zi(card)
			res_cards = list(set(card).difference(p_lz))
			p_lz.sort()
			res_cards.sort(key=Rules.value_and_suit, reverse=True)
			if len(p_lz) == len(Rules.lz_cards()):
				if Rules.value(res_cards[-1]) == 1:
					return [p_lz[-1], res_cards[1], p_lz[0], res_cards[0]]
				return [p_lz[-1], res_cards[0], p_lz[0], res_cards[1]]
			else:
				if Rules.value(res_cards[0]) == Rules.value(res_cards[1]):
					return [res_cards[0], res_cards[1], p_lz[0], res_cards[-1]]
				if Rules.value(res_cards[-1]) == 1:
					return_cards = res_cards[1:]
					return_cards.extend(p_lz)
					return_cards.append(res_cards[0])
					return return_cards
				res_cards.insert(0, p_lz[0])
				return res_cards
		elif sa_pu_op == SA_PU_SY_TX:
			card.sort(key=Rules.value_and_suit, reverse=True)
			return card
		lz_cards = Rules.lz_cards()
		if Rules.compare_one_by_lai_zi(card[:2], card[2:], lz_cards) == IS_MORE:
			big_card, small_card = card[:2], card[2:]
		else:
			big_card, small_card = card[2:], card[:2]

		b_card = Rules.compare_one_card(big_card)
		s_card = Rules.compare_one_card(small_card)
		return b_card + s_card

	@staticmethod
	def sort_big_small(card, sa_pu_op=SA_PU_LP):
		if not card:
			return
		if Rules.check_has_lai_zi(card):
			return Rules.sort_big_small_by_lai_zi(card, sa_pu_op)
		if sa_pu_op != SA_PU_SY:
			big_card, small_card = Rules.search_bigger_small_combine(card)
		else:
			card.sort(key=Rules.value(card))
			big_card, small_card = Rules.search_bigger_small_combine(card)
		b_card = Rules.compare_one_card(big_card)
		s_card = Rules.compare_one_card(small_card)
		return b_card + s_card

	@staticmethod
	def sort_big_small_cards(cards):
		"""
		排序大小扑
		"""
		if not cards:
			return
		b_card = Rules.compare_one_card(cards[2:])
		s_card = Rules.compare_one_card(cards[2:])
		return b_card + s_card