# -*- coding: utf-8 -*-

from copy import deepcopy

from ...const.mahjong.const import HuPaiType, ActionType, CardType

from ..mahjong.base import RuleBase

class RuleMahjong(RuleBase):
	"""
	麻将规则
	"""
	@staticmethod
	def get_count_list_by_value(cards):
		"""
		统计整组牌张数量
		"""
		count_list = {}
		for v in cards:
			count_list[v] = count_list.get(v, 0) + 1
		return count_list

	@staticmethod
	def search_pairs(cards):
		"""
		搜索可组成的对子列表
		"""
		count_list = RuleMahjong.get_count_list_by_value(cards)
		result = list()
		for card, count in count_list.items():
			if count >= 2:
				result.append(card)
		return result

	@staticmethod
	def search_count(cards, zhi_ding_count):
		"""
		搜索可组成对子牌的列表
		"""
		count_list = RuleMahjong.get_count_list_by_value(cards)
		result = list()
		for card, count in count_list.items():
			if count == zhi_ding_count:
				result.append(card)
		if result:
			result.sort()
		return result

	@staticmethod
	def search_cards_by_count(cards, count1, count2, count3):
		"""
		搜索满足指定count的牌，用于代替search_count, 一次变量搜索即可
		"""
		count_list = RuleMahjong.get_count_list_by_value(cards)
		cards_1, cards_2, cards_3 = [], [], []
		for card, count in count_list.items():
			if count == count1:
				cards_1.append(card)
			elif count == count2:
				cards_2.append(card)
			elif count == count3:
				cards_3.append(card)
		return cards_1, cards_2, cards_3

	@staticmethod
	def calc_ke_zi_list_and_must_list(cards):
		"""
		计算刻子列表，以及必须被满足的牌列表
		"""
		count_list = RuleMahjong.get_count_list_by_value(cards)
		ke_zi_list = list()
		must_list = list()
		for value, count in count_list.items():
			if count == 1:
				must_list.append(value)
			elif count == 2:
				must_list.append(value)
				must_list.append(value)
			elif count == 3:
				ke_zi_list.append(value)
			elif count == 4:
				must_list.append(value)
				ke_zi_list.append(value)
		return ke_zi_list, must_list

	@staticmethod
	def is_seven_pairs(cards, lai_zi):
		"""
		TODO: 判断是否为小七对， 只需考虑一个癞子内的情况
		手中由7副对子组成的牌型
		"""
		# 当玩家手牌不等于14张时，直接返回，不进行小七对判断
		if not cards or len(cards) != 14:
			return False, []
		cards = list(cards)
		lai_zi_count = RuleMahjong.remove_by_value(cards, lai_zi, -1)
		count_list = RuleMahjong.get_count_list_by_value(cards)
		singles = [card for card, count in count_list.items() if count == 1]
		threes = [card for card, count in count_list.items() if count == 3]
		fours = [card for card, count in count_list.items() if count == 4]
		singles_len = len(singles)
		threes_len = len(threes)
		# 当癞子数量为0时
		if lai_zi_count == 0:
			if threes_len == 0 and singles_len == 0:
				result = HuPaiType.QI_DUI
				if len(fours) > 0:
					result = HuPaiType.LONG_QI_DUI
				return result, [[card] * count for card, count in count_list.items()]
		# 当癞子数量大于0时(但是每一位玩家手中最多只有一张癞子牌)
		elif lai_zi_count > 0 and singles_len + threes_len <= lai_zi_count:  # 8对 + 癞子  一刻子 + 6对 + 1单牌 +癞子
			singles_path = []
			singles_path.extend([[card] * count for card, count in count_list.items() if count == 2 or count == 4])
			singles_path.extend([[card] * 4 for card, count in count_list.items() if count == 3])
			singles_path.extend([[card] * 2 for card, count in count_list.items() if count == 1])

			# 玩家胡牌类型为七对
			result = HuPaiType.QI_DUI
			# 当存在四张或三张相同卡牌时，则为龙七对
			if len(fours) > 0 or threes_len > 0:
				result = HuPaiType.LONG_QI_DUI

			return result, singles_path

		return False, []

	@staticmethod
	def is_da_dui_zi(cards, lai_zi):
		"""
		TODO: 判断是否为大队子
		4副刻子(三张相同的牌)加上1个对子
		"""
		# 4 X 3 + 2 = 14
		# 判断玩家手牌是否为14张，组合后剩余对子牌
		if not cards or len(cards) % 3 != 2:
			return False, []
		cards = list(cards)
		# 统计癞子数量
		lai_zi_count = RuleMahjong.remove_by_value(cards, lai_zi, -1)
		# 统计卡牌数量: {13: 3, 11: 2, ....}
		count_list = RuleMahjong.get_count_list_by_value(cards)
		# 牌型: 单张、对子、刻子，杠
		singles, two, threes, fours = [], [], [], []
		for card, count in count_list.items():
			if count == 1:
				singles.append(cards)
			elif count == 2:
				two.append(card)
			elif count == 3:
				threes.append(card)
			elif count == 4:
				fours.append(card)

		# 当玩家癞子数量为0时，说明玩家将癞子牌打出
		# 当单张和四张都不存在，并且对子数量为1，则为大队子
		if lai_zi_count == 0:
			if not singles and not fours and len(two) == 1:
				return HuPaiType.DA_DUI_ZI, [[card] * count for card, count in count_list.items()]
		# 当玩家手中持有癞子牌，并且不存在四张相同的牌
		elif lai_zi_count > 0 and not fours:
			singles_path = []
			first_pair = False
			# 如果存在对子牌
			if two:
				# 单张需要癞子，对子需要一张癞子，当有对子时，留出一对可减少一张癞子
				if len(singles) * 2 + (len(two) - 1) <= lai_zi_count:
					for card, count in count_list.items():
						if count == 1:
							singles_path.append([card] * 3)
						elif count == 2:
							if first_pair:
								singles_path.append([card] * 2)
							else:
								singles_path.append([card] * 3)
						elif count == 3:
							singles_path.append([card] * 3)
					# 大队子
					return HuPaiType.DA_DUI_ZI, singles_path
			# 如果玩家手中没有对子牌
			else:
				if len(singles) * 2 - 1 <= lai_zi_count:
					for card, count in count_list.items():
						if count == 1:
							if first_pair:
								singles_path.append([card] * 2)
							else:
								singles_path.append([card] * 3)
						elif count == 3:
							singles_path.append([card] * 3)
					return HuPaiType.DA_DUI_ZI, singles_path
		return False, []

	@staticmethod
	def is_di_long_qi(piles, cards, lai_zi):
		"""
		TODO: 判断是否是地龙七, 休闲场中地龙七算龙七对
		手牌由5个对子和4张相同的牌组成
		"""
		# 必须先有一个碰牌  这个在外部直接PASS
		# 必须有一个碰牌+摸或接炮同一张牌才继续内部判断
		# 这里只需要判断剩下的是5对即可
		# 最多有一个杠或者碰
		if len(piles) != 1 or piles[0] != ActionType.PONG:
			return False, []
		# 地龙七手上必须有11张
		if not cards or len(cards) != 11:
			return False, []
		cards = list(cards)
		cards.extend(piles[1])
		lai_zi_count = RuleMahjong.remove_by_value(cards, lai_zi, -1)
		count_list = RuleMahjong.get_count_list_by_value(cards)
		singles = [card for card, count in count_list.items() if count == 1]
		threes = [card for card, count in count_list.items() if count == 3]
		fours = [card for card, count in count_list.items() if count == 4]
		singles_len = len(singles)
		threes_len = len(threes)
		if lai_zi_count == 0:
			if threes_len == 0 and singles_len == 0:
				result = False
				if len(fours) == 1:
					result = HuPaiType.LONG_QI_DUI
				return result, [[card] * count for card, count in count_list.items()]

		# 癞子为1(因为玩家手中最多只有一张癞子)时
		# 8对 + 癞子  一刻子 + 6对 + 1单牌 +癞子
		elif lai_zi_count > 0 and singles_len + threes_len == lai_zi_count:
			singles_path = []
			singles_path.extend([[card] * count for card, count in count_list.items() if count == 2 or count == 4])
			singles_path.extend([[card] * 4 for card, count in count_list.items() if count == 3])
			singles_path.extend([[card] * 2 for card, count in count_list.items() if count == 1])
			result = False
			if len(fours) == 1 or threes_len == 1:
				result = HuPaiType.LONG_QI_DUI
			return result, singles_path

		return False, []

	@staticmethod
	def is_di_qi_dui(cards, lai_zi):
		"""
		判断是否7小对 只需考虑1个癞子内的情况
		"""
		# 必须先有一个碰牌  这个在外部直接PASS  必须有一个碰牌+摸或接炮同一张牌才继续内部判断 这里只需要判断剩下的是5对即可
		if not cards or len(cards) != 10:
			return False, []
		cards = list(cards)
		lai_zi_count =  RuleMahjong.remove_by_value(cards, lai_zi, -1)
		count_list = RuleMahjong.get_count_list_by_value(cards)
		singles = [card for card, count in count_list.items() if count == 1]
		threes = [card for card, count in count_list.items() if count == 3]
		fours = [card for card, count in count_list.items() if count == 4]
		singles_len = len(singles)
		threes_len = len(threes)
		if lai_zi_count == 0:
			if threes_len == 0 and singles_len == 0:
				result = HuPaiType.SEVER_PAIR
				if len(fours) > 0:
					result = HuPaiType.SEVER_PAIR_HAO_HUA
				return result, [[card] * count for card, count in count_list.items()]
		elif lai_zi_count > 0 and singles_len + threes_len <= lai_zi_count:  # 8对 + 癞子  一刻子 + 6对 + 1单牌 +癞子
			singles_path = []
			singles_path.extend([[card] * count for card, count in count_list.items() if count == 2 or count == 4])
			singles_path.extend([[card] * 4 for card, count in count_list.items() if count == 3])
			singles_path.extend([[card] * 2 for card, count in count_list.items() if count == 1])
			result = HuPaiType.SEVER_PAIR
			if len(fours) > 0 or threes_len > 0:
				result = HuPaiType.SEVER_PAIR_HAO_HUA
			return result, singles_path

		return False, []

	@staticmethod
	def is_hh_seven_pairs(cards, lai_zi):
		"""
		判断是否豪华7小对 只需考虑2个癞子内的情况，3金倒分数比七对高，不考虑  刻字+七对
		"""
		if not cards or len(cards) != 14:
			return False, []
		cards = list(cards)
		lai_zi_count = RuleMahjong.remove_by_value(cards, lai_zi, -1)
		count_list = RuleMahjong.get_count_list_by_value(cards)
		singles = [card for card, count in count_list.items() if count == 1]
		threes = [card for card, count in count_list.items() if count == 3]
		singles_len = len(singles)
		threes_len = len(threes)
		if lai_zi_count == 0:
			if threes_len == 0 and singles_len == 0:
				return True, [[card] * count for card, count in count_list.items()]
		elif lai_zi_count > 0 and singles_len + threes_len <= lai_zi_count:  # 8对 + 癞子  一刻子 + 6对 + 1单牌 +癞子
			singles_path = []
			singles_path.extend([[card] * count for card, count in count_list.items() if count == 2 or count == 4])
			singles_path.extend([[card] * count for card, count in count_list.items() if count == 3])
			singles_path.extend([[card] * count for card, count in count_list.items() if count == 1])
			singles_path.append([lai_zi] * lai_zi_count)
			return True, singles_path

		return False, []

	@staticmethod
	def is_group_match_rule(cards):
		"""
		TODO: 判断牌值的分组是否符合麻将的顺子、刻子的规则
		"""
		cards_len = len(cards)
		hu_path = []
		if cards_len == 0:
			return True, hu_path

		cards.sort()
		if cards_len == 3:
			# 判断牌值是否全部由顺子构成，注意这里不能直接传牌过来，只能传牌值，不能带花色
			# 所有会改变原参数的值的方法，都应该在开始的时候直接复制list
			is_shun_zi, path = RuleBase.is_value_shun_zi(cards)
			# 判断牌值是否为刻子，注意，这里不能直接传牌过去，只能传牌值，不能带花色
			if is_shun_zi or RuleBase.is_value_ke_zi(cards):
				hu_path.append(cards)
				return True, hu_path

			return False, []

		# 计算刻子列表，以及必须被满足的牌列表
		ke_zi_list, must_list = RuleMahjong.calc_ke_zi_list_and_must_list(cards)
		hu_path = list(map(lambda value: [value] * 3, ke_zi_list))
		if len(must_list) == 0:
			return True, hu_path

		if len(ke_zi_list) == 0:
			flag, shun_zi_path = RuleBase.is_value_shun_zi(cards)
			hu_path.extend(shun_zi_path)
			return flag, hu_path

		must_list.sort()
		is_shun_zi, shun_zi_path = RuleBase.is_value_shun_zi(must_list)
		if is_shun_zi:
			hu_path.extend(shun_zi_path)
			return True, hu_path

		ke_zi_list.sort()
		for v in ke_zi_list:
			tmp_value = list(must_list)
			tmp_value.extend([v] * 3)
			tmp_value.sort()
			is_shun_zi, shun_zi_path = RuleBase.is_value_shun_zi(tmp_value)
			if is_shun_zi:
				hu_path.extend(shun_zi_path)
				return True, hu_path

		for v in ke_zi_list:
			tmp_value = list(cards)
			tmp_value.remove(v)
			tmp_value.remove(v)
			tmp_value.remove(v)
			tmp_value.sort()
			is_shun_zi, shun_zi_path = RuleBase.is_value_shun_zi(tmp_value)
			if is_shun_zi:
				hu_path.extend(shun_zi_path)
				return True, hu_path

		tmp_value = list(cards)
		tmp_value.sort()
		is_shun_zi, shun_zi_path = RuleBase.is_value_shun_zi(tmp_value)
		if is_shun_zi:
			hu_path.extend(shun_zi_path)
			return True, hu_path

		return False, []

	@staticmethod
	def can_hu_by_jiang(cards, card):
		"""
		TODO: 判断能否以此为将牌胡牌
		"""
		cards = list(cards)
		remove_jiang_count = 2
		RuleMahjong.remove_by_value(cards, card, remove_jiang_count)
		group = RuleBase.group_by_suit(cards)
		hu_path = []
		for k, v in group.items():
			if len(v) % 3 != 0:
				return False, []
			flag, unit_hu_path = RuleMahjong.is_group_match_rule(list(v))
			for path in unit_hu_path:
				hu_path.append(list(map(lambda value: k * 10 + value, path)))
			if not flag:
				return False, []

		hu_path.append([card] * remove_jiang_count)
		return True, hu_path

	@staticmethod
	def __can_hu_with_pairs_and_jiang(cards, lai_zi_count, remove_jiang, lai_zi=CardType.LAI_ZI):
		"""
		TODO: 玩家胡对或将
		"""
		# pair_list = RuleMahjong.search_pairs(cards)
		# two_list = RuleMahjong.search_count(cards, 2)
		# one_list = RuleMahjong.search_count(cards, 1)
		# three_list = RuleMahjong.search_count(cards, 3)
		# TODO: 统计玩家1张牌的数量，2张牌的数量，3张牌的数量
		one_list, two_list, three_list = RuleMahjong.search_cards_by_count(cards, 1, 2, 3)
		# 判断是否存在将
		for jiang in two_list:
			flag, hu_path = RuleMahjong.can_hu_with_lai_zi_and_jiang(cards, jiang, lai_zi_count, remove_jiang, lai_zi)
			if flag:
				return True, hu_path
		# 还差将牌
		for jiang in one_list:
			flag, hu_path = RuleMahjong.can_hu_with_lai_zi_and_jiang(cards, jiang, lai_zi_count, remove_jiang, lai_zi)
			if flag:
				return True, hu_path
		# 将牌在三张中
		for jiang in three_list:
			flag, hu_path = RuleMahjong.can_hu_with_lai_zi_and_jiang(cards, jiang, lai_zi_count, remove_jiang, lai_zi)
			if flag:
				return True, hu_path
		# 从卡牌中寻找将牌
		for jiang in cards:
			flag, hu_path = RuleMahjong.can_hu_with_lai_zi_and_jiang(
				cards, jiang, lai_zi_count - 1, remove_jiang - 1, lai_zi)
			if flag:
				return True, hu_path

		return False, []

	@staticmethod
	def can_hu_without_lai_zi(cards, _):
		"""
		TODO: 不带赖子判断胡牌
		"""
		pair_list = RuleMahjong.search_pairs(cards)
		for card in pair_list:
			hu, path = RuleMahjong.can_hu_by_jiang(cards, card)
			if hu:
				return True, path

		return False, []

	@staticmethod
	def can_hu_with_one_lai_zi(cards, lai_zi):
		"""
		一个癞子判断胡牌
		手里有将，则先尝试用将牌组合，判断能否胡
		如果没有将，则直接尝试红中补将
		"""
		# 一个癞子判断胡牌
		RuleMahjong.remove_by_value(cards, lai_zi, -1)
		# 能胡对或将
		return RuleMahjong.__can_hu_with_pairs_and_jiang(cards, 1, 2, lai_zi)

	@staticmethod
	def can_hu(cards, lai_zi=CardType.LAI_ZI):
		"""
		判断胡牌的总循环
		"""
		#
		if len(cards) % 3 != 2:
			return False, []
		cards = list(cards)
		# TODO: 只存在两种情况[癞子打出(0)，癞子没打(1)]
		method_map = {
			# is_hu: True, False
			0: RuleMahjong.can_hu_without_lai_zi,
			1: RuleMahjong.can_hu_with_one_lai_zi,
		}
		# hz_count = [0, 1]
		# method_map.get(hz_count):
		#   1.选择 -> [0, 1, 2, 3, 4]
		#   2.返回 -> (is_hu: True, False, path: hu_cards)
		#   3.参数: (cards, lai_zi) -> (cards, lai_zi)是作为参数传入method_map中取到的方法

		# 统计·癞子数量
		lz_count = RuleBase.calc_value_count(cards, CardType.LAI_ZI)

		# 判断胡牌操作
		if method_map.get(lz_count):
			is_hu, hu_path = method_map.get(lz_count)(cards, lai_zi)
			if is_hu:
				hu_path = list(filter(lambda v: v != [], hu_path))
			return is_hu, hu_path

		return False, []

	@staticmethod
	def get_hu_path(suit, step_data, lai_zi):
		"""
		判断胡的牌
		"""
		step_data = deepcopy(step_data)
		hu_path = []
		user_cards = []
		for step in reversed(step_data):
			for already_user_card in user_cards:
				if already_user_card in step[3]:
					step[3].remove(already_user_card)

			user_cards.extend(step[3])
			cards = list(map(lambda v: suit * 10 + v, step[3]))
			hz_cards = list(map(lambda v: suit * 10 + v, step[4]))
			unit_path = (cards + hz_cards)
			unit_path.sort()
			for value in step[4]:
				index = unit_path.index(suit * 10 + value)
				unit_path[index] = lai_zi
			hu_path.append(unit_path)

		return hu_path

	@staticmethod
	def can_hu_with_lai_zi_and_jiang(cards, jiang, lai_zi_count, remove_jiang_count, lai_zi=CardType.LAI_ZI):
		"""
		TODO: 能胡癞子和将
		"""
		cards = list(cards)
		RuleMahjong.remove_by_value(cards, jiang, remove_jiang_count)
		# 麻将分组[1: 11, 13, 15,...], [2: 21, 22, 24,...]
		group = RuleBase.group_by_suit(cards).items()
		group = sorted(group, key=lambda value: len(value[1]), reverse=True)
		hu_path = []
		for suit, card_list in group:
			flag, lai_zi_count_temp, step_data = RuleMahjong.check_value_match_rule_with_lai_zi_count(
				list(card_list), 0, list(), lai_zi_count, False)
			if not flag:
				return False, []
			else:
				hu_path.extend(RuleMahjong.get_hu_path(suit, step_data, lai_zi))
			lai_zi_count = lai_zi_count_temp

		hu_path.append([jiang] * remove_jiang_count)
		for value in hu_path:
			if len(value) == 1:
				value.append(lai_zi)

		return True, hu_path

	@staticmethod
	def check_value_is_valid_with_lai_zi(cards, step_data, index, hz_count):
		"""
		:param cards:
		:param step_data:
		:param index:
		:param hz_count:
		:return: 是否组成顺子或刻子, 当前牌, 红中数量，红中所变的牌
		"""
		calcCards = list(cards)
		value = cards[0]
		hz_change_value = []
		if not step_data[index][1]:
			step_data[index][1] = True
			count = RuleBase.calc_value_count(cards, value)
			if 3 <= count:
				RuleMahjong.remove_by_value(cards, value, 3)
				return True, cards, hz_count, hz_change_value

			if hz_count > 0 and hz_count >= 3 - count:
				RuleMahjong.remove_by_value(cards, value, 3)
				hz_count -= (3 - count)
				hz_change_value.extend([value] * (3 - count))
				return True, cards, hz_count, hz_change_value
		if not step_data[index][0]:
			step_data[index][0] = True
			if value + 1 in cards and value + 2 in cards:
				RuleMahjong.remove_by_value(cards, value)
				RuleMahjong.remove_by_value(cards, value + 1)
				RuleMahjong.remove_by_value(cards, value + 2)
				return True, cards, hz_count, hz_change_value

			used_hz = 0
			if value + 1 not in cards:
				used_hz += 1
				hz_change_value.append(value + 1)

			if value + 2 not in cards:
				used_hz += 1
				if value + 2 <= 9:
					hz_change_value.append(value + 2)
				else:
					hz_change_value.append(value - 1)

			if hz_count >= used_hz > 0:
				hz_count -= used_hz
				RuleMahjong.remove_by_value(cards, value)
				RuleMahjong.remove_by_value(cards, value + 1)
				if value + 2 <= 9:
					RuleMahjong.remove_by_value(cards, value + 2)
				else:
					RuleMahjong.remove_by_value(cards, value - 2)
				return True, cards, hz_count, hz_change_value
			else:
				hz_change_value = []

		return False, calcCards, hz_count, hz_change_value

	@staticmethod
	def check_value_match_rule_with_lai_zi_count(cards, index, step_data, hz_count, is_back):
		"""
		检测某花色是否符合游戏规则（带红中检测）
		从最左边的牌往右依次来检测，当成顺或成刻时，此路通
		当即不成顺又不成刻时，此路不通
		当此路通时，往下循环，到达终点时则是成功
		当此路不通时，往上回退一步，如果上一步已经检测了顺和刻，则再回退一步，
		直到可以选择下一步或者回到了起点。
		"""
		if len(cards) == 0:
			return True, hz_count, step_data,

		cards.sort()
		if index >= len(step_data):
			step_data.append([False, False, 0, {}, []])

		if not is_back:
			step_data[index][2] = hz_count
			step_data[index][3] = deepcopy(cards)

		flag, new_list, new_hz_count, new_hz_change_value = RuleMahjong.check_value_is_valid_with_lai_zi(
			cards, step_data, index, hz_count)

		# 7
		step_data[index][4] = new_hz_change_value
		if flag:
			return RuleMahjong.check_value_match_rule_with_lai_zi_count(
				new_list, index + 1, step_data, new_hz_count, False)

		if index > 0:
			step_data = step_data[0:index]
			old_hz_count, old_list = deepcopy(step_data[index - 1][2]), deepcopy(step_data[index - 1][3])

			return RuleMahjong.check_value_match_rule_with_lai_zi_count(
				old_list, index - 1, step_data, old_hz_count, True)

		if index == 0:
			return False, hz_count, step_data

		return False, hz_count, step_data