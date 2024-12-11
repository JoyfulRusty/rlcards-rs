# -*- coding: utf-8 -*-

from rlcards.const.sytx import const
from rlcards.games.sytx.rules import Rules
from rlcards.const.sytx.const import LandLordAction, FarmerAction


class SyRound:
	"""
	水鱼流程
	"""
	def __init__(self, np_random, dealer, players):
		"""
		初始化水鱼流程参数
		"""
		self.dealer = dealer
		self.players = players
		self.np_random = np_random
		self.winner_id = None
		self.last_action = None
		self.curr_player_id = None
		self.last_player_id = None

	def proceed_round(self, action):
		"""
		todo: 强攻、密、撒扑流程操作
		"""
		# todo: 闲家强攻
		if action == FarmerAction.QG:
			return self.compare_qg(self.curr_player_id, action)

		# todo: 闲家密
		elif action == FarmerAction.MI or self.last_action == FarmerAction.MI:
			return self.compare_mi(self.curr_player_id, action)

		# todo: 闲家撒扑，庄家选择操作
		else:
			return self.compare_sa_pu(self.curr_player_id, action)

	@staticmethod
	def select_next_id(p_id):
		"""
		下一位操作ID
		"""
		if p_id == 1:
			return 0
		return 1

	def get_state(self, curr_p, other_hand_cards, hand_card_nums, predict_infos, round_actions):
		"""
		获取状态
		"""
		state = {
			"actions": self.get_actions(curr_p, round_actions),
			"self": curr_p.player_id,
			"role": curr_p.role,
			"is_winner": curr_p.is_winner,
			"curr_hand_cards": curr_p.curr_hand_cards,
			"other_hand_cards": other_hand_cards,
			"hand_card_nums": hand_card_nums,
			"last_action": self.last_action,
			"round_actions": round_actions,
			"predict_infos": predict_infos,
			"remain_cards": self.get_remain_cards(),
			"used_by_cards": self.get_used_by_cards(curr_p, other_hand_cards)
		}

		return state

	def get_remain_cards(self):
		"""
		解码未使用卡牌
		"""
		remain_cards = []
		for card in self.dealer.deck:
			remain_cards.append(card.get_card_value())
		return remain_cards

	@staticmethod
	def get_used_by_cards(curr_player, other_hand_cards):
		"""
		已使用卡牌
		"""
		used_by_cards = other_hand_cards + curr_player.curr_hand_cards

		return used_by_cards

	def get_actions(self, curr_p, round_actions):
		"""
		获取动作
		"""
		# 区分闲/庄
		# 判断对局是否结束
		if self.winner_id is not None:
			return [self.last_action]
		# 根据玩家角色选择对应合法操作
		choice_actions = {
			"landlord": self.get_landlord_actions(),
			"farmer": self.get_farmer_actions(round_actions)
		}
		return choice_actions[curr_p.role]

	def get_landlord_actions(self):
		"""
		庄家判断选择动作
		"""
		# 闲家操作完后，庄家判断选择操作
		if self.last_action == FarmerAction.MI:
			return [LandLordAction.REN, LandLordAction.KAI]
		elif self.last_action == FarmerAction.SP:
			return [LandLordAction.SHA, LandLordAction.ZOU]

	def get_farmer_actions(self, round_actions):
		"""
		闲家判断选择动作
		"""
		# 开局闲家选择动作
		if not round_actions and not self.last_action:
			return [FarmerAction.SP, FarmerAction.QG, FarmerAction.MI]
		# 闲家判断操作[信、反]
		return [FarmerAction.XIN, FarmerAction.FAN]

	def calc_action_reward(self, lose_id, win_id, action):
		"""
		分别设置赢家和输家奖励
		"""
		init_reward = 0.2  # 初始化奖励
		# 强攻
		if action == FarmerAction.QG:
			dynamics_reward = init_reward + 0.1
		# 开牌
		elif action == LandLordAction.KAI:
			dynamics_reward = init_reward + 0.2
		# 反
		elif action == FarmerAction.FAN:
			dynamics_reward = init_reward + 0.15
		# 认、信
		else:
			dynamics_reward = init_reward

		self.players[win_id].action_reward = dynamics_reward
		self.players[lose_id].action_reward = -dynamics_reward

	def calc_extra_reward(self, lose_id, win_id, reward):
		"""
		计算额外奖励
		"""
		self.players[win_id].extra_reward += reward
		self.players[lose_id].extra_reward -= reward

	def calc_cards_by_reward(self):
		"""
		当前卡牌值设置奖励计算
		"""
		for curr_p in self.players:
			curr_p.cards_reward = self.split_big_and_small_by_cards(curr_p.curr_hand_cards)

	def split_big_and_small_by_cards(self, cards: list):
		"""
		拆分大小扑
		"""
		# 卡牌边界值
		bound = 5
		# 大扑与小扑
		big_cards = cards[:2]
		small_cards = cards[2:]

		# 判断癞子和豹子
		big_has_lz = self.calc_has_by_lai_zi(big_cards)
		small_has_lz = self.calc_has_by_lai_zi(small_cards)
		big_has_pairs = self.calc_has_by_pairs(big_cards)
		small_has_pairs = self.calc_has_by_pairs(small_cards)

		# 1.处理设置存在癞子牌时，奖励设置
		if big_has_lz and small_has_lz:
			return 0.025 * bound

		elif big_has_lz:
			return 0.02 * bound + (self.calc_split_attack_cards_by_sum(small_cards) - bound) * 0.05

		elif big_has_pairs and small_has_pairs:
			return 0.02 * bound

		elif big_has_pairs:
			return 0.01 * bound + (self.calc_split_attack_cards_by_sum(small_cards) - bound) * 0.05
		# 癞子与豹子均不存在
		else:
			# 计算大扑与小扑值大小
			big_value = self.calc_split_attack_cards_by_sum(big_cards)
			small_value = self.calc_split_attack_cards_by_sum(small_cards)
			if not small_value:
				if big_value >= bound and small_value >= bound:
					return 0.05 * bound

				elif big_value >= bound and small_value <= bound:
					return 0.01 * bound + (small_value - bound) * 0.05

				elif big_value <= bound and small_value <= bound:
					return 0.01 * bound + ((big_value - bound) * 0.075) + ((small_value - bound) * 0.075)

				elif big_value <= bound and small_value >= bound:
					return 0.01 * bound + ((big_value - bound) * 0.075)
			else:
				return -(0.05 * bound) + ((big_value - bound) * 0.075)

	@staticmethod
	def calc_has_by_lai_zi(cards: list):
		"""
		判断是否存在癞子
		"""
		# 判断当前卡牌是否存在癞子
		if set(cards) & set(const.LAI_ZI):
			return True
		return False

	@staticmethod
	def calc_has_by_pairs(cards: list):
		"""
		判断是否存在豹子[对子]
		"""
		# 判断当前卡牌是否存在对子[豹子]
		if len(set(cards)) == 1:
			return True
		return False

	@staticmethod
	def calc_split_attack_cards_by_sum(cards: list):
		"""
		计算分扑卡牌值
		"""
		tmp_values = []
		for card in cards:
			# 判断花牌值，花牌的值为零
			if card % 100 > 9:
				tmp_values.append(0)
			# 当不是花牌时，则计算卡牌值大小
			else:
				tmp_values.append(card % 10)

		# 判断当前卡牌扑组合值是否大于10
		# 当前卡牌扑组合值最大只能为9
		if sum(tmp_values) < 10:
			return sum(tmp_values)
		else:
			return sum(tmp_values) % 10

	def compare_qg(self, curr_id, action):
		"""
		todo: 闲家首次选择操作为强攻时
			1.庄家只能选择开，比牌大小判定输赢家

		强攻分析:
			1.闲家强攻，庄家必开
			2.庄家两扑比闲家大，闲家赔2倍
			3.闲家两扑比庄家大，庄家赔2倍
		"""
		# 记录上一个动作
		self.last_action = action
		self.last_player_id = curr_id
		# 闲、庄当前手牌
		farmer_cards, landlord_cards = self.calc_compare_cards()
		# 比较卡牌大小，根据输出结果来判定输赢家
		res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_X_QG_Z_KAI_NUM)
		# todo: 区分闲家强攻时当前手牌是否具有大牌或嘿鸡，闲家小牌选择强攻
		# [庄家为赢家]
		if not res:
			# 1.计算赢家奖励
			self.winner_id = self.select_next_id(curr_id)
			self.players[self.winner_id].is_winner = True
			self.calc_action_reward(curr_id, self.winner_id, action)

			# 2.计算大小扑奖励
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [大小扑其中一扑大]
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, self.winner_id, 0.1)

			# [大小扑其中一扑小]
			if big_res == const.IS_LESS or small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.1)

			# [大小扑都小]
			if big_res == const.IS_LESS and small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.2)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

		# [闲家为赢家]
		else:
			# 1.计算赢家奖励
			self.winner_id = curr_id
			self.players[self.winner_id].is_winner = True
			lose_id = self.select_next_id(curr_id)
			self.calc_action_reward(lose_id, curr_id, action)

			# 2.计算大小扑奖励
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [大小扑都大]
			if big_res == const.IS_MORE and small_res == const.IS_MORE:
				self.calc_extra_reward(lose_id, self.winner_id, 0.2)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

	def compare_mi(self, curr_id, action):
		"""
		todo: 闲家首次操作为密时
			1.庄家选择: 认 —> 庄家输
			2.庄家选择: 开 -> 比牌大小判定输赢家

		密牌分析:
			1.庄家不看牌，赔1倍
			2.庄家看牌
				2.1闲家两扑比庄家大，庄家赔2倍
				2.2庄家两扑大于闲家，闲家赔2倍
		"""
		# 记录上一个动作
		self.last_action = action
		self.last_player_id = curr_id
		# 闲密庄开，直接比较卡牌大小，根据比较结果来判断输赢家
		if action == LandLordAction.KAI:
			farmer_cards, landlord_cards = self.calc_compare_cards()
			res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_X_MI_Z_KAI_NUM)
			# [庄家为赢家]
			if not res:
				# 1.计算赢家奖励
				self.winner_id = curr_id
				self.players[self.winner_id].is_winner = True
				lose_id = self.select_next_id(curr_id)
				self.calc_action_reward(lose_id, self.winner_id, action)

				# 2.计算大小扑奖励
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# [闲家大小扑都小]
				if big_res == const.IS_LESS and small_res == const.IS_LESS:
					self.calc_extra_reward(lose_id, self.winner_id, 0.2)

				# [闲家大小扑其中一个小]
				if big_res == const.IS_LESS or small_res == const.IS_LESS:
					self.calc_extra_reward(lose_id, self.winner_id, 0.1)

				# [闲家大小扑其中一个大]
				if big_res == const.IS_MORE or small_res == const.IS_MORE:
					self.calc_extra_reward(lose_id, self.winner_id, 0.1)

				# 3.计算本局卡牌值奖励
				self.calc_cards_by_reward()

			# [闲家为赢家]
			else:
				# 1.计算赢家奖励
				self.winner_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(curr_id, self.winner_id, action)

				# 2.计算大小扑奖励
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# [闲家大小扑都大]
				if big_res == const.IS_MORE and small_res == const.IS_MORE:
					self.calc_extra_reward(curr_id, self.winner_id, 0.2)

				# 3.计算本局卡牌值奖励
				self.calc_cards_by_reward()

		# [闲家为赢家]
		elif action == LandLordAction.REN:
			# 1.计算赢家奖励
			self.winner_id = self.select_next_id(curr_id)
			self.players[self.winner_id].is_winner = True
			self.calc_action_reward(curr_id, self.winner_id, action)

			# 2.计算大小扑奖励
			farmer_cards, landlord_cards = self.calc_compare_cards()
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [闲家大小扑都大]
			if big_res == const.IS_LESS and small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.25)

			# [闲家大小扑其中一个小]
			if big_res == const.IS_LESS or small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.1)

			# [闲家大小扑其中一个大]
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, self.winner_id, 0.1)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

	def compare_sa_pu(self, curr_id, action):
		"""
		todo: 闲家首次操作为撒扑时
			1.庄家选择：杀 -> 闲家: 信、反
			2.庄家选择：走 -> 闲家: 信、反

		撒扑分析:
			1.庄家走闲家信，无赢家，平局
			2.庄家认，闲家为赢家
			3.庄家杀
				3.1闲家信，赔1倍
				3.2闲家不信
					a: 庄家大两扑，闲家赔2倍
					b: 庄家两扑小，庄家赔2倍
		"""
		# todo: 闲撒 -> 庄杀 -> 闲信
		if action == FarmerAction.XIN and self.last_action == LandLordAction.SHA:
			# [庄家为赢家]
			# 1.计算赢家奖励
			self.winner_id = self.select_next_id(curr_id)
			self.players[self.winner_id].is_winner = True
			self.calc_action_reward(curr_id, self.winner_id, action)

			# 2.计算大小扑奖励
			farmer_cards, landlord_cards = self.calc_compare_cards()
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [闲家大小扑都小]
			if big_res == const.IS_LESS and small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.25)

			# [闲家大小扑其中一扑小]
			if big_res == const.IS_LESS or small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.15)

			# [闲家大小扑都小]
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, self.winner_id, 0.15)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

		# todo: 闲撒 -> 庄杀 -> 闲反(比两扑)
		elif action == FarmerAction.FAN and self.last_action == LandLordAction.SHA:
			farmer_cards, landlord_cards = self.calc_compare_cards()
			res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_Z_SHA_X_FAN_NUM)
			# todo: 庄杀闲反
			# [庄家为赢家]
			if not res:
				# 1.计算赢家奖励
				self.winner_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(curr_id, self.winner_id, action)

				# 2.计算大小扑奖励
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# [闲家大小扑都小]
				if big_res == const.IS_LESS and small_res == const.IS_LESS:
					self.calc_extra_reward(curr_id, self.winner_id, 0.2)

				# [闲家大小扑其中一扑小]
				if big_res == const.IS_LESS or small_res == const.IS_LESS:
					self.calc_extra_reward(curr_id, self.winner_id, 0.15)

				# [闲家大小扑其中一扑大]
				if big_res == const.IS_MORE or small_res == const.IS_MORE:
					self.calc_extra_reward(curr_id, self.winner_id, 0.15)

				# 3.计算本局卡牌值奖励
				self.calc_cards_by_reward()

			# [闲家为赢家]
			else:
				# 1.计算赢家奖励
				self.winner_id = curr_id
				self.players[self.winner_id].is_winner = True
				lose_id = self.select_next_id(curr_id)
				self.calc_action_reward(lose_id, self.winner_id, action)

				# 2.计算大小扑奖励
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# [闲家大小扑都大]
				if big_res == const.IS_MORE and small_res == const.IS_MORE:
					self.calc_extra_reward(curr_id, self.winner_id, 0.2)

				# 3.计算本局卡牌值奖励
				self.calc_cards_by_reward()

		# todo: 闲撒 -> 庄走 -> 闲信，无赢家
		elif action == FarmerAction.XIN and self.last_action == LandLordAction.ZOU:
			# 庄走闲信，[无赢家]
			self.winner_id = -1
			tmp_win_id = self.select_next_id(curr_id)
			farmer_cards, landlord_cards = self.calc_compare_cards()

			# 1.计算大小扑奖励
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [闲家大小扑都大]
			if big_res == const.IS_MORE and small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, tmp_win_id, 0.2)

			# [闲家大小扑都小]
			if big_res == const.IS_LESS and small_res == const.IS_LESS:
				self.calc_extra_reward(tmp_win_id, curr_id, 0.25)

			# [闲家大小扑其中一扑小]
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, tmp_win_id, 0.1)

			# [闲家大小扑都小]，庄家可赢，但选择了走
			if big_res == const.IS_LESS or small_res == const.IS_LESS:
				self.calc_extra_reward(self.winner_id, curr_id, 0.1)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

		elif action == FarmerAction.FAN and self.last_action == LandLordAction.ZOU:
			# todo: 闲撒 -> 庄走 -> 闲反(比一扑)
			farmer_cards, landlord_cards = self.calc_compare_cards()
			res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_Z_ZOU_X_FAN_NUM)
			# todo: 庄走闲反
			# [庄家为赢家]
			if not res:
				# 1.计算赢家奖励
				self.winner_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(curr_id, self.winner_id, action)

				# 2.计算大小扑奖励
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# [闲家大小扑其中一扑小]
				if big_res == const.IS_LESS or small_res == const.IS_LESS:
					self.calc_extra_reward(curr_id, self.winner_id, 0.1)

				# [闲家大小扑其中一扑大]
				if big_res == const.IS_MORE or small_res == const.IS_MORE:
					self.calc_extra_reward(curr_id, self.winner_id, 0.1)

				# [闲家大小扑都小]
				if big_res == const.IS_LESS and small_res == const.IS_LESS:
					self.calc_extra_reward(self.winner_id, curr_id, 0.25)

				# 3.计算本局卡牌值奖励
				self.calc_cards_by_reward()

			# [闲家为赢家]
			else:
				# 1.计算赢家奖励
				self.winner_id = curr_id
				self.players[self.winner_id].is_winner = True
				lose_id = self.select_next_id(curr_id)
				self.calc_action_reward(lose_id, self.winner_id, action)

				# 2.计算大小扑奖励
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# [闲家大小扑都大]
				if big_res == const.IS_MORE and small_res == const.IS_MORE:
					self.calc_extra_reward(lose_id, self.winner_id, 0.2)

				# 3.计算本局卡牌值奖励
				self.calc_cards_by_reward()

		# 记录上一个动作
		self.last_action = action
		# 判断出赢家后，将不再更新上一位玩家ID
		if not self.winner_id:
			self.last_player_id = curr_id

	def check_sy_tx(self, curr_id):
		"""
		todo: is_bigger判断比较结果
			1.res == False: 庄家赢
			2.res == True: 闲家赢

		当前闲家出现出现水鱼或水鱼天下
		则直接进入比牌流程，输出赢家
		"""
		farmer_cards, landlord_cards = self.calc_compare_cards()
		x_sy_tx = Rules.is_sy_tx(farmer_cards)  # 水鱼天下 [110, 110, 110, 110]
		x_sy = Rules.is_sy(farmer_cards)  # 水鱼 [110, 110, 119, 119]
		if x_sy_tx or x_sy:
			# 庄闲为水鱼或水鱼天下
			if self.calc_landlord_sy_tx(landlord_cards):
				res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_Z_AND_X_SY)
				# 庄家对象为赢家
				if not res:
					self.winner_id = self.select_next_id(curr_id)
				# 闲家对象为赢家
				else:
					self.winner_id = curr_id
			# 仅闲家为水鱼或水鱼天下
			else:
				res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_X_SY)
				# 庄家对象为赢家
				if not res:
					self.winner_id = self.select_next_id(curr_id)
				# 闲家对象为赢家
				else:
					self.winner_id = curr_id

	def calc_compare_cards(self):
		"""
		统计当前卡牌值分布，用于计算输赢
		"""
		farmer_cards = []
		landlord_cards = []
		for player in self.players:
			# [闲家卡牌]
			if player.role == "farmer":
				farmer_cards.extend(player.curr_hand_cards)
			# [庄家卡牌]
			else:
				landlord_cards.extend(player.curr_hand_cards)
		return farmer_cards, landlord_cards

	@staticmethod
	def calc_landlord_sy_tx(cards):
		"""
		计算庄家水鱼或水鱼天下
		"""
		z_sy_tx = Rules.is_sy_tx(cards)
		z_sy = Rules.is_sy(cards)
		if z_sy or z_sy_tx:
			return True

		return False