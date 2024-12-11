# -*- coding: utf-8 -*-

from rlcards.utils.mahjong.xxc import RuleXXC
from rlcards.const.mahjong.const import ActionType, HuPaiType


class MahjongPlayer:
	"""
	红中麻将捡漏玩家
	"""
	def __init__(self, player_id, np_random):
		"""
		初始化玩家属性参数
		"""
		self.is_out = False  							# 玩家破产
		self.call_pai = 0                               # 玩家是否叫牌
		self.lock_cards = []  							# 玩家锁住的牌
		self.all_chu_cards = []  						# 玩家打出所有操作
		self.curr_hand_cards = []						# 玩家手牌, []
		self.np_random = np_random                      # 随机种子
		self.player_id = player_id                      # 玩家ID

		self.piles = []  								# 存储玩家碰、杠操作历史
		self.operates = []  							# 存储玩家可选动作

		self.tian_hu = False                            # 玩家天胡
		self.tian_ting = False  						# 玩家天听(状态, 玩家选择之后设置为True)
		self.can_tian_ting = False                      # 玩家是否能天听
		self.bi_men_yi_shou = False                     # 必须闷一次才能胡开

		self.ting_list = []  							# 玩家天听卡牌
		self.ting_infos = []                            # 玩家听牌信息

		self.men_card_infos = []  						# 玩家闷牌信息
		self.pick_card_infos = []  						# 玩家捡牌信息
		self.hu_infos = []  							# 玩家胡牌信息

		self.pong_infos = []  							# 玩家碰牌信息
		self.gang_infos = []  							# 玩家杠牌信息(明杠、索杠、暗杠、憨包杠)
		self.gang_hou_mo_card = []  					# 玩家杠后摸牌
		self.gang_hou_chu_card = []                     # 玩家杠后出牌
		self.sj_gang_seats = set()  					# 烧鸡杠位置玩家
		self.han_bao_gang_count = 0  					# 统计玩家憨包杠
		self.suo_gang_state = False                     # 玩家处于杠牌状态中

		# 记录牌
		self.ze_ren_ji = 0  							# 责任鸡
		self.ze_ren_wgj = 0                             # 责任乌骨鸡
		self.wg_ji = 0  								# 乌骨鸡(2)
		self.cf_ji = 0  								# 冲锋鸡
		self.cf_wg_ji = 0  								# 冲锋乌骨鸡(4)
		self.cf_jwg_ji = 0  							# 冲锋金乌骨鸡(8)
		self.wg_zr_ji = 0  								# 责任乌骨鸡(4)
		self.jwg_ji = 0  								# 金乌乌骨鸡(4)
		self.zr_jwg_ji = 0  							# 责任金乌骨鸡(8)
		self.fk_ji_cards = []  							# 玩家手中的鸡牌

		self.rewards = 0.0                       		# 每位玩家动作奖励(例如: 被杠-0.1, 杠+0.1)
		self.lz = 0                      		        # 连庄次数

	def remove_cards(self, card, count=0):
		"""
		删除操作牌(碰、杠)
		"""
		for _ in range(count):
			self.curr_hand_cards.remove(card)

	def calc_suo_gang(self, mo_card):
		"""
		TODO: 检查当前摸牌之前是否产生了碰牌
		"""
		for i in range(len(self.piles)):
			# 卡牌和类型
			action_type = self.piles[i][0]
			card_value = self.piles[i][1]

			# 判断碰操作
			if action_type != ActionType.PONG:
				continue

			# 判断摸牌是否与碰牌相等
			if card_value[0] != mo_card:
				continue

			# 当前摸牌在碰牌中
			if mo_card in card_value:
				# 删除玩家手牌之前存储的碰操作
				self.piles.remove(self.piles[i])
				return True

		return False

	def out_of_lock_cards(self):
		"""
		计算玩家锁牌之后能打出的牌
		"""
		# [:]赋值手牌
		can_play_card = self.curr_hand_cards[:]

		# 将已经锁住的牌删除，剩余卡牌则为可出卡牌
		for card in self.lock_cards:
			can_play_card.remove(card)

		return can_play_card

	def check_permit(self, hu_type):
		"""
		检查胡牌权限: 通行证
		"""
		# 已操作天听、杠，则不需要判断是否符合通行证
		if self.tian_ting:
			return False
		if self.gang_infos or self.men_card_infos or self.pick_card_infos:
			return False

		# 胡牌类型不是平胡，则返回，通行证检查，仅限于平胡
		if hu_type != HuPaiType.PING_HU:
			return False

		# 判断平胡是否符合通行证胡牌条件
		permits = [ActionType.MING_GONG, ActionType.AN_GONG, ActionType.SUO_GONG]
		for action in self.piles:
			if action[0] in permits:
				return True

		return False

	def pong_de_qi(self, last_card):
		"""
		判断玩家能否碰得起
		"""
		# 天听状态不能碰牌
		if self.tian_ting:
			return False, 0

		# 处于锁牌状态则不能碰牌
		if self.lock_cards:
			return False, 0

		# 计算玩家碰
		is_pong, card = RuleXXC.can_pong(self.curr_hand_cards, last_card)
		if is_pong:
			self.operates.append(ActionType.PONG)  # [(方位，动作，卡牌)]
			return is_pong, card

		return False, 0

	def ming_gang_de_qi(self, last_card):
		"""
		判断玩家能否明杠
		"""
		# 天听状态不能明杠
		if self.tian_ting:
			return False, 0
		# 锁牌状态不能明杠
		if self.lock_cards:
			return False, 0
		# 计算玩家明杠
		is_ming_gang, card = RuleXXC.can_ming_gang(self.curr_hand_cards, last_card)
		if is_ming_gang:
			self.operates.append(ActionType.MING_GONG)  # [(方位，动作，卡牌)]
			return is_ming_gang, card

		return False, 0

	def suo_gang_de_qi(self, mo_card):
		"""
		判断玩家能否索杠
		"""
		# 计算玩家是否能转弯杠(索杠)
		is_suo_gang, card = RuleXXC.can_zhuan_wan_gang(self.curr_hand_cards, self.piles, mo_card)
		if is_suo_gang:
			self.operates.append(ActionType.SUO_GONG)  # [(方位，动作，卡牌)]
			return is_suo_gang, card

		return False, 0

	def an_gang_de_qi(self, mo_card):
		"""
		判断玩家能否暗杠
		"""
		# 锁牌状态下不能暗杠
		if self.lock_cards:
			return False, 0
		# 计算玩家暗杠
		is_an_gang, card = RuleXXC.can_an_gang(self.curr_hand_cards, mo_card)
		if is_an_gang:
			self.operates.append(ActionType.AN_GONG)  # [(方位，动作，卡牌)]
			return is_an_gang, card

		return False, 0

	# TODO: 玩家天听
	def can_select_tian_ting_14(self):
		"""
		玩家(仅)第一次摸牌，计算当前手牌是否能天听
		"""
		# 检查是否符合天听条件
		if len(self.curr_hand_cards) != 14:
			return False
		# 判断天听
		if len(self.piles) == 0 and not self.can_tian_ting:
			if not self.all_chu_cards:
				hu_type = RuleXXC.r_can_tian_ting_14(self.piles, self.curr_hand_cards)
				if hu_type:
					return hu_type

		return False

	def can_tian_ting_after_first_an_gang(self):
		"""
		判断玩家未出过牌，但暗杠一次后摸牌能不能天听
		"""
		# 判断所有的出牌是否为4张
		if len(self.all_chu_cards) != 4:
			return False, 0
		# 暗杠后手牌剩余10张+补牌1张，此时手牌为11张
		if len(self.curr_hand_cards) != 11:
			return False, 0
		if len(self.piles) != 1:
			return False, 0
		# 判断类型是否为暗杠
		for z in self.piles:
			if z[0] != ActionType.AN_GONG:
				return False, 0
		if self.tian_ting > True:
			return RuleXXC.can_tian_ting_after_first_an_gang(self.piles, self.curr_hand_cards)

		return False, 0

	def calc_stand_ji(self, default_ji):
		"""
		计算站鸡，站鸡仅限默认鸡，冲锋鸡为打出去的鸡没有站鸡
		在手牌中，自摸的，捡胡的，暗杠的幺鸡和乌骨鸡分数翻倍(可与金鸡相乘)
		"""
		# 手牌
		stand_ji = []
		for card in self.curr_hand_cards:
			if card in default_ji:
				stand_ji.append(card)

		# 胡开的牌
		for hu_info in self.hu_infos:
			card = hu_info.get("hu_card")
			if card in default_ji:
				stand_ji.append(card)

		# 闷牌
		for men_data in self.men_card_infos:
			card = men_data["men_card"]
			if card in default_ji:
				stand_ji.append(card)

		# 闷牌
		for pick_data in self.pick_card_infos:
			card = pick_data["pick_card"]
			if card in default_ji:
				stand_ji.append(card)

		# 暗杠的牌
		for combo in self.piles:
			if combo[0] != ActionType.AN_GONG:
				continue
			for card in combo[1]:
				if card in default_ji:
					stand_ji.append(card)

		return stand_ji

	def calc_all_ji_cards(self, default_ji, dealer, fan_ji_list=None, with_out=False):
		"""
		计算自己所有鸡牌
		"""
		fan_ji_list = fan_ji_list or set()
		all_bird = default_ji | fan_ji_list  # 默认鸡+翻鸡牌

		# 手牌
		for card in self.curr_hand_cards:
			if card in all_bird:
				self.fk_ji_cards.append(card)

		# 胡开的牌
		for hu_info in self.hu_infos:
			card = hu_info.get("hu_card")
			if card in default_ji:
				self.fk_ji_cards.append(card)

		# 闷牌
		for men_data in self.men_card_infos:
			card = men_data["men_card"]
			if card in all_bird:
				self.fk_ji_cards.append(card)

		# 捡牌
		for pick_data in self.pick_card_infos:
			card = pick_data["pick_card"]
			if card in all_bird:
				self.fk_ji_cards.append(card)

		# 碰牌
		for combo in self.piles:
			for card in combo[1]:
				if card in all_bird:
					self.fk_ji_cards.append(card)

		# 默认鸡
		for card in dealer.table:
			if card in default_ji:
				self.fk_ji_cards.append(card)

		# 出过的牌(满堂鸡)
		if with_out:
			for card in dealer.table:
				if card in default_ji:  # 打出默认鸡已经算过一遍
					continue
				if card in fan_ji_list:
					self.fk_ji_cards.append(card)

	def clear_operates(self):
		"""
		清空已存在有效操作
		"""
		self.operates.clear()