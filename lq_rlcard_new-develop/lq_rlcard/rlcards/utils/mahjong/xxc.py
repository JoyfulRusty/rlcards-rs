# -*- coding: utf-8 -*-

from collections import defaultdict

from copy import deepcopy

from ...const.mahjong import const
from ...const.mahjong.const import ActionType, HuPaiType, CardType, ExtraHuPaiType

from ..mahjong.base import RuleBase
from ..mahjong.mahjong import RuleMahjong


class RuleXXC(RuleBase):

	@staticmethod
	def get_hu_type(curr_player, players, last_player_id, landlord_id, card, is_zi_mo=False):
		"""
		TODO: 此接口计算玩家手牌胡牌类型，玩家能不能胡可不用考虑
		玩家选择胡开，分两种情况:
		    1.自摸胡开
		    2.点炮胡开
		胡流程分析:
		1.如果玩家胡牌类型为平胡，则需要进通行证检查
		2.当玩家出牌时，符合其他玩家胡牌类型，称为点炮
			a: 对其他两位玩家检查是否进行过捡/密，是否符合胡牌，如果玩家只有杠(无密和捡)，则该操作为平胡
			b: 如果其他两位玩家进行过密和捡(包含杠)，如果能胡开，则不需要进行通行证检查
		3.自摸选择开(自己摸牌胡，如果没有进行过密胡，无杠，需要检查通行证)
		"""
		# 判断玩家是否能胡及胡牌类型及胡的卡牌
		hu_type, hu_path = RuleXXC.can_hu(curr_player.piles, curr_player.curr_hand_cards, card)
		if not hu_type:
			return False, []

		# 小七对、大对子、龙七对、清一色
		is_seven_pair = hu_type == HuPaiType.QI_DUI
		is_pong_hu = hu_type == HuPaiType.DA_DUI_ZI
		is_hh_seven_pair = hu_type == HuPaiType.LONG_QI_DUI
		is_qing_yi_se = RuleXXC.has_hu_is_qing_yi_se(
			deepcopy(curr_player.piles),
			deepcopy(curr_player.curr_hand_cards)
		)

		# 天湖对于庄而言，天湖后的庄点炮也没有其他额外分
		is_tian_hu = False
		# 判断当前玩家ID是否等于庄家ID
		if curr_player.player_id == landlord_id:
			if not curr_player.all_chu_cards and curr_player.tian_hu:
				is_tian_hu = True

		is_sha_bao = False  # 判断杀报和天胡
		bei_sha_bao_seats = []  # 被杀报位置
		get_hu_type = []  # 胡牌类型(正常胡牌和额外胡牌)

		# 自摸
		if is_zi_mo:
			# 杠上花
			if len(curr_player.gang_hou_mo_card) > 0:
				curr_player.rewards += 0.1
				# 计算其他玩家奖励
				for other_player in players:
					if other_player == curr_player:
						continue
					other_player.rewards -= 0.1
				get_hu_type.append(ExtraHuPaiType.GS_HUA)

			for p_id, other_player in enumerate(players):
				if not other_player:
					continue
				if other_player== curr_player:
					continue
				if other_player == players[landlord_id]:  # 庄家不能天听
					continue
				if other_player.tian_ting:
					is_sha_bao = True
					bei_sha_bao_seats.append(other_player.player_id)
		# 点炮
		else:
			# 抢杠胡(其他玩家杠，杠牌可抢胡)
			if players[last_player_id].suo_gang_state:
				curr_player.rewards += 0.1
				players[last_player_id] -= 0.1
				get_hu_type.append(ExtraHuPaiType.QG_HU)
				curr_player.sj_gang_seats.add(last_player_id)
				players[last_player_id].suo_gang_state = False

			# 杠上炮(玩家杠后放炮)
			if len(players[last_player_id].gang_hou_chu_card) > 0:
				curr_player.rewards += 0.1
				players[last_player_id].rewards -= 0.1
				get_hu_type.append(ExtraHuPaiType.GS_PAO)
				curr_player.sj_gang_seats.add(last_player_id)

			# 地胡
			if not curr_player.all_chu_cards and not curr_player.men_card_infos and curr_player.player_id == landlord_id:
				get_hu_type.append(ExtraHuPaiType.DI_HU)

			# 杀报(报听玩家未胡牌)
			if curr_player.player_id != landlord_id and curr_player.tian_ting:
				is_sha_bao = True
				bei_sha_bao_seats.append(curr_player.player_id)

		# 判断清一色
		if is_qing_yi_se:
			get_hu_type.append(HuPaiType.QING_YI_SE)			# 清一色
			if is_hh_seven_pair:
				get_hu_type.append(HuPaiType.LONG_QI_DUI)		# 清龙七对
			elif is_seven_pair:
				get_hu_type.append(HuPaiType.QING_QI_DUI)		# 清七对
			elif is_pong_hu:
				get_hu_type.append(HuPaiType.QING_DA_DUI)		# 清大对
		# 当不是清一色，则判断其他类型
		elif is_hh_seven_pair:
			get_hu_type.append(HuPaiType.LONG_QI_DUI)
		elif is_seven_pair:
			get_hu_type.append(HuPaiType.QI_DUI)
		elif is_pong_hu:
			get_hu_type.append(HuPaiType.DA_DUI_ZI)
		else:
			get_hu_type.append(HuPaiType.PING_HU)

		# 不包括以上胡牌类型
		# 庄家没有天听
		is_tian_ting = curr_player.player_id != landlord_id and curr_player.tian_ting
		is_tian_hu and get_hu_type.append(ExtraHuPaiType.TIAN_HU)  # 天胡
		is_tian_ting and get_hu_type.append(ExtraHuPaiType.TIAN_TING)  # 天听
		is_sha_bao and get_hu_type.append(ExtraHuPaiType.SHA_BAO)  # 杀报

		return get_hu_type, hu_path

	@staticmethod
	def can_hu(piles, cards, card, allow_seven=True, lz=const.LAI_ZI):
		"""
		TODO: 计算是否能胡牌
		"""
		# 红中不允许吃胡
		if card == const.LAI_ZI:
			return False, []

		def lz_2_hz(value):
			if value == lz:
				return const.LAI_ZI
			return value

		def hz_2_lz(value):
			if value == const.LAI_ZI:
				return const.LAI_ZI
			return value

		if lz != const.LAI_ZI:
			cards = list(map(lz_2_hz, cards))
		cards = list(cards)

		if RuleBase.is_card(card) and len(cards) % 3 != 2:
			cards.append(card)

		# 判断小七对胡牌
		if allow_seven:
			'''
			flag: hu_type -> 胡牌类型
			path: hu_cards -> 能胡的卡牌
			'''
			hu_type, hu_path = RuleMahjong.is_seven_pairs(cards, CardType.LAI_ZI)
			if hu_type:
				return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 判断大队子胡牌
		hu_type, hu_path = RuleMahjong.is_da_dui_zi(cards, lz)
		if hu_type:
			return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 判断龙七对胡牌
		# 休闲场中: 地龙七算龙七对
		hu_type, hu_path = RuleMahjong.is_di_long_qi(piles, cards, lz)
		if hu_type:
			return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 判断胡的总循环
		hu_type, hu_path = RuleMahjong.can_hu(cards, lz)
		if hu_type:
			return HuPaiType.PING_HU, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 如果不能胡
		return False, []

	@staticmethod
	def calc_can_tian_ting(piles, hand_cards):
		"""
		TODO: 计算玩家当前手牌是否满足天听条件
		"""
		cards = deepcopy(hand_cards)
		tmp_card = None
		for card in cards:
			if tmp_card == card:
				continue
			tmp_card = card
			tian_ting_hand_cards = deepcopy(cards)
			tian_ting_hand_cards.remove(card)

			# 玩家打出一张牌之后就能听牌
			hu_type, _ = RuleXXC.can_ting_pai(piles, tian_ting_hand_cards)
			if hu_type:
				return hu_type

		return False

	@staticmethod
	def r_can_tian_ting_14(piles, hand_cards):
		"""
		TODO: 是否可以天听，此接口使用，必须在桌子里面判断没出过牌
		"""

		return RuleXXC.calc_can_tian_ting(piles, hand_cards)

	@staticmethod
	def r_can_tian_ting_13(piles, hand_cards, allow_seven=True, lai_zi=const.LAI_ZI):
		"""
		TODO: 是否可以天听，此接口使用时，必须在桌子里面先判断没出过牌
		"""
		for combo in piles:
			if combo[0] != ActionType.AN_GONG:
				return False
		cards = deepcopy(hand_cards)
		# 玩家打出一张即可天听
		hu_type, hu_path = RuleXXC.can_ting_pai(piles, cards, allow_seven, lai_zi)
		if hu_type:
			return hu_type, hu_path
		return False, []

	@staticmethod
	def can_tian_ting_after_first_an_gang(piles, hand_cards):
		"""
		TODO: 计算起手牌暗杠之后，能否听牌
		"""
		can_ting, hu_path = RuleXXC.calc_can_tian_ting(piles, hand_cards)
		return can_ting, hu_path

	@staticmethod
	def get_round_over_call_cards(piles, hand_cards, curr_card=0, allow_seven=True, allow_131=False):
		"""
		此接口获取玩家叫牌类型，外部不再出来
		"""
		cards = deepcopy(hand_cards)
		# 如果有14张，打出一张之后算法听牌
		if len(hand_cards) % 3 == 2:
			cards_list = cards[:]
			cards_list.sort()
			curr_c = 0
			for card in cards_list:
				if card == curr_c:
					continue
				tian_ting_hand_cards = deepcopy(hand_cards)
				tian_ting_hand_cards.remove(card)
				# 计算听牌
				is_hu, hu_path = RuleXXC.can_ting_pai(piles, tian_ting_hand_cards, allow_seven)
				if is_hu:
					return RuleXXC.get_call_type(piles, is_hu, hu_path, cards, curr_card)
		else:
			is_hu, hu_path = RuleXXC.can_ting_pai(piles, cards, allow_seven)
			if is_hu:
				return RuleXXC.get_call_type(piles, is_hu, hu_path, cards, curr_card)
		return 0

	@staticmethod
	def get_call_type(piles, can_hu, hu_path, cards, curr_card):
		"""
		在手牌形成叫牌的基础上计算叫牌类型
		如: 清一色|大队子|小七对等
		"""
		# 叫牌类型
		call_type = HuPaiType.PING_HU
		is_seven_pair = 0  # 七对
		is_hh_seven_pairs = 0  # 龙七对
		is_pong_pong_hu = 0  # 大对子
		is_di_long_qi = 0  # 地龙七
		# 判断七对
		if can_hu == HuPaiType.QI_DUI:
			is_seven_pair = 1
			call_type = HuPaiType.QI_DUI
			list_3 = RuleXXC.get_card_list_by_count(cards, 3, True)
			if len(list_3) > 0:
				is_hh_seven_pairs = 1
				call_type = HuPaiType.LONG_QI_DUI
		if can_hu == HuPaiType.DI_LONG_QI:
			is_hh_seven_pairs = 1
			call_type = HuPaiType.DI_LONG_QI
		# 判断碰胡
		if is_seven_pair == 0:
			is_pong_pong_hu = RuleXXC.is_pong_pong_hu(deepcopy(hu_path), deepcopy(piles))
			if is_pong_pong_hu:
				call_type = HuPaiType.DA_DUI_ZI
		# 判断清一色
		is_qing_yi_se = RuleXXC.has_hu_is_qing_yi_se(deepcopy(piles), cards)
		if is_qing_yi_se:
			if is_seven_pair:
				call_type = HuPaiType.QING_QI_DUI
			elif is_hh_seven_pairs:
				call_type = HuPaiType.QING_LONG_QI
			elif is_pong_pong_hu:
				call_type = HuPaiType.QING_DA_DUI
			elif is_di_long_qi:
				call_type = HuPaiType.QING_DI_LONG
			else:
				call_type = HuPaiType.QING_YI_SE

		return call_type

	@staticmethod
	def get_tian_ting_cards(piles, cards, allow_seven=True, lai_zi=const.LAI_ZI):
		"""
		TODO: 是否可以天听，以及打出那几张能天听
		注意: 需要在桌子里先判断没出过牌
		"""
		# 玩家听牌列表
		tian_ting_cards = []
		cards_copy = deepcopy(cards)
		# 牌型卡牌列表
		cards_copy.sort()
		if len(cards) == 13:
			hu_type, hu_path = RuleXXC.can_ting_pai(piles, cards_copy, allow_seven, lai_zi)
			if hu_type:
				return True, tian_ting_cards

		# 计算玩家打出之后才能听牌
		c = 0
		for card in cards_copy:
			if c == card:
				continue
			c = card
			if card in tian_ting_cards:
				tian_ting_cards.append(card)
				continue
			tian_ting_hand_cards = deepcopy(cards)
			tian_ting_hand_cards.remove(card)
			# 打出之后能听牌
			hu_type, hu_path = RuleXXC.can_ting_pai(piles, tian_ting_hand_cards, allow_seven, lai_zi)
			if hu_type:
				tian_ting_cards.append(card)

		# 当玩家听牌大于0时
		if len(tian_ting_cards) > 0:
			return True, tian_ting_cards

		return False, []

	@staticmethod
	def get_ting_pai_cards_and_hu_list(cards, allow_seven=True, allow_131=False, lai_zi=const.LAI_ZI):
		""" 获取听牌和胡牌列表 """
		tian_ting_cards = []
		for card in cards:
			tian_ting_hand_cards = deepcopy(cards)
			tian_ting_hand_cards.remove(card)
			# 打出一张牌之后能听牌
			ting_list = RuleXXC.get_ting_hu_list(tian_ting_hand_cards, allow_seven, allow_131)
			if len(ting_list) > 0:
				tian_ting_cards.append([card, ting_list])

			if len(tian_ting_cards) > 0:
				return True, tian_ting_cards

		return False, []

	@staticmethod
	def can_ting_pai(piles, hand_cards, allow_seven=True, lai_zi=const.LAI_ZI):
		"""
		TODO: 计算是否为听牌类型和所听卡牌
		判断玩家听牌的胡牌类型
		"""
		def lz_2_hz(value):
			if value == lai_zi:
				return const.LAI_ZI
			return value

		def hz_2_lz(value):
			if value == const.LAI_ZI:
				return lai_zi
			return value

		cards = deepcopy(hand_cards)
		if len(cards) % 3 < 2:
			cards.append(lai_zi)

		if lai_zi != const.LAI_ZI:
			cards = list(map(lz_2_hz, hand_cards))

		# 判断玩家是否胡牌小七对
		if allow_seven:
			hu_type, hu_path = RuleMahjong.is_seven_pairs(cards, const.LAI_ZI)
			if hu_type:
				return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 判断玩家是否胡牌大队子
		hu_type, hu_path = RuleMahjong.is_da_dui_zi(cards, const.LAI_ZI)
		if hu_type:
			return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 判断玩家是否胡牌地龙七
		hu_type, hu_path = RuleMahjong.is_di_long_qi(piles, cards, lai_zi)
		if hu_type:
			return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		# 判断玩家胡
		hu_type, hu_path = RuleMahjong.can_hu(cards, const.LAI_ZI)
		if hu_type:
			return hu_type, list(map(lambda v: list(map(hz_2_lz, v)), hu_path))

		return False, []

	@staticmethod
	def get_ting_hu_list(hand_cards, allow_seven=True, lai_zi=const.LAI_ZI):
		""" 获取听牌列表 """
		def lz_2_hz(value):
			if value == lai_zi:
				return const.LAI_ZI
			return value

		if lai_zi != const.LAI_ZI:
			hand_cards = list(map(lz_2_hz, hand_cards))

		hu_path = []
		count_gt_4 = RuleMahjong.search_count(hand_cards, 4)
		for card in const.ALL_CARDS:
			if card in count_gt_4:
				continue
			cards = list(hand_cards)
			cards.append(card)
			# 是否为小七对
			if allow_seven:
				is_hu, path_cards = RuleMahjong.is_seven_pairs(cards, const.LAI_ZI)
				if is_hu:
					hu_path.append(card)
					continue
			# 判断是否能胡
			is_hu, path_cards = RuleMahjong.can_hu(cards, const.LAI_ZI)
			if is_hu:
				hu_path.append(card)
				continue
		return hu_path

	@staticmethod
	def has_hu_is_qing_yi_se(cheng_cards, cards):
		""" 清一色 """
		# 当前摸牌不等于0时，则添加此牌到玩家手牌中
		# if card > 0:
		# 	cards.append(card)
		# # for pile_cards in cheng_cards:
		# # 	cards.extend(pile_cards[1])
		# # 计算当前玩家手牌中的癞子数量
		# lai_zi_count = RuleBase.calc_value_count(cards, lai_zi)
		# if cheng_cards:
		# 	print("输出cheng_cards: ", cheng_cards)
		# 	cards = reduce(lambda result, v: result.extend(v[1:-1]), cheng_cards, cards)
		# RuleMahjong.remove_by_value(cards, const.LAI_ZI, lai_zi_count)
		# if cards:
		# 	hua_se = set(map(lambda value: value // 10, cards))
		# 	return len(hua_se) == 1
		# return False
		for pile_cards in cheng_cards:
			cards.extend(pile_cards[1])

		tmp_hand_cards = cards[:]
		cards_by_type = defaultdict(list)
		for card in tmp_hand_cards:
			# 处理手牌中的癞子牌
			if card == CardType.LAI_ZI:
				cards.remove(card)
			_type = card // 10
			_value = card % 10
			cards_by_type[_type].append(_value)

		# 统计当前手牌类型是否只存在一种花色
		calc_type = list(cards_by_type.keys())

		# 大于1，则不是清一色
		if len(calc_type) > 1:
			return False

		return True

	@staticmethod
	def is_pong_pong_hu(hu_path, piles):
		""" 碰碰胡 """
		for combo in hu_path:
			if len(combo) > 1:
				combo_set = set(combo)
				if len(combo_set) > 2:
					return False
				if len(combo_set) == 2 and const.LAI_ZI not in combo_set:
					return False
		return True

	@staticmethod
	def can_ming_gang(cards: list, card):
		""" 能明杠 """
		if card == const.LAI_ZI:
			return False, 0
		if RuleBase.calc_value_count(cards, card) >= 3:
			return True, card
		return False, 0

	@staticmethod
	def can_zhuan_wan_gang(cards: list, piles: list, card=0):
		""" 能转弯杠 """
		cards = cards[:]
		if card != 0:
			for i in range(len(piles)):
				action_type = piles[i][0]
				first_card = piles[i][1]
				if action_type != ActionType.SUO_GONG:
					continue
				if first_card != card:
					continue
				return True, card

		for card in cards:
			for i in range(len(piles)):
				action_type = piles[i][0]
				first_card = piles[i][1]
				if action_type != ActionType.SUO_GONG:
					continue
				if first_card != card:
					continue
				return True, card

		return False, 0

	@staticmethod
	def can_an_gang(cards: list, card=0):
		""" 能暗杠 """
		if card > 0:
			if RuleBase.calc_value_count(cards, card) >= 4:
				return True, card
		for card in cards:
			if RuleBase.calc_value_count(cards, card) >= 4:
				return True, card
		return False, 0

	@staticmethod
	def can_pong(cards: list, card):
		""" 能碰 """
		if card == const.LAI_ZI:
			return False, 0
		if RuleBase.calc_value_count(cards, card) >= 2:
			return True, card
		return False, 0