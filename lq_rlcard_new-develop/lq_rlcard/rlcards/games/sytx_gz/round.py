# -*- coding: utf-8 -*-

import random

from rlcards.const.sytx_gz import const
from rlcards.games.sytx_gz.rules import Rules
from rlcards.const.sytx_gz.const import LandLordAction, FarmerAction


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

		贵州水鱼撒扑特殊判断:
			1.庄杀闲信，闲输
			2.庄走闲信，平局
			3.闲密庄认，庄输
			以上三种情况均不打开暗牌

		除此之外情况均打开暗牌比较大小

		打开暗牌时，若任意一方存在水鱼或水鱼天下则直接比较牌型大小
		1.强攻必开时，比较大小扑: 2
		2.密牌时，庄家选择开，比较大小扑: 2
		3.庄杀闲反，比较大小扑: 1
		4.庄走闲反，比较大小扑: 2
		"""
		# todo: 闲家强攻
		if action == FarmerAction.QG:
			return self.compare_qg(self.curr_player_id, action)

		# todo: 闲家密
		elif action == FarmerAction.MI or self.last_action == FarmerAction.MI:
			return self.compare_mi(self.curr_player_id, action)

		# todo: 庄家撒扑，闲家也进行撒扑
		else:
			return self.compare_sa_pu(self.curr_player_id, action)

	@staticmethod
	def select_next_id(p_id):
		"""
		下一位操作ID
		"""
		return 0 if p_id == 1 else 1

	def get_state(self, curr_p, other_hand_cards, round_actions, history_actions):
		"""
		获取状态
		"""
		state = {
			"actions": self.get_actions(curr_p, round_actions),
			"role": curr_p.role,
			"is_winner": curr_p.is_winner,
			"bright_cards": curr_p.bright_cards,
			"combine_cards": curr_p.combine_cards,
			"compare_big_cards": self.compare_farmer_and_landlord_big_cards(round_actions),
			"other_hand_cards": other_hand_cards,
			"last_action": self.last_action,
			"round_actions": round_actions,
			"history_actions": history_actions,
			"remain_cards": self.get_remain_cards(),
		}

		return state

	def calc_big_or_small_res(self, round_actions):
		"""
		todo: 闲家撒扑时，比较亮牌大扑大小
		"""
		# 判断当前
		if not round_actions:
			return None, None
		farmer_big_cards = []
		landlord_big_cards = []
		# 计算闲家撒扑时，比较庄闲家已亮牌大扑大小
		if round_actions[0][1] == FarmerAction.SP:
			for curr_p in self.players:
				if curr_p.role == "farmer":
					farmer_big_cards.extend(curr_p.combine_cards[:2])
				elif curr_p.role == "landlord":
					landlord_big_cards.extend(curr_p.combine_cards[:2])
			return farmer_big_cards, landlord_big_cards
		return None, None

	def compare_farmer_and_landlord_big_cards(self, round_actions):
		"""
		todo: 比较闲家亮牌时，庄闲组合的大扑大小
		"""
		farmer_big_cards, landlord_big_cards = self.calc_big_or_small_res(round_actions)
		if not farmer_big_cards and not landlord_big_cards:
			return False
		farmer_big_res = Rules.compare_one_combine(farmer_big_cards, landlord_big_cards)
		# 庄家家大扑大
		if farmer_big_res == 4:
			return "landlord"
		# 闲家大扑大
		elif farmer_big_res == 3:
			return "farmer"
		# 相等
		return "draw"

	def get_remain_cards(self):
		"""
		解码未使用卡牌
		"""
		return [card.get_card_value() for card in self.dealer.deck]

	def get_actions(self, curr_p, round_actions):
		"""
		获取动作
		"""
		# 区分闲/庄
		# 判断对局是否结束
		if self.winner_id is not None:
			return [self.last_action]

		# 闲家合法动作选择
		elif curr_p.role == "farmer":
			return self.get_farmer_actions(curr_p, round_actions)

		# 庄家合法动作选择
		elif curr_p.role == "landlord":
			return self.get_landlord_actions()

	def get_landlord_actions(self):
		"""
		庄家判断选择动作
		"""
		# todo: 根据大扑组成牌值，添加庄家动作选择判断
		if self.last_action == FarmerAction.MI:
			return [LandLordAction.REN, LandLordAction.KAI]
		elif self.last_action == FarmerAction.SP:
			return [LandLordAction.SHA, LandLordAction.ZOU]

	def get_farmer_actions(self, curr_p, round_actions):
		"""
		闲家判断选择动作
		"""
		# todo: 根据大扑组成牌值，添加闲家动作选择判断
		# 开局闲家选择动作
		if not round_actions and not self.last_action:
			return self.calc_qg_and_mi(curr_p)
		# 闲家判断操作[信、反]
		return [FarmerAction.XIN, FarmerAction.FAN]

	def calc_qg_and_mi(self, curr_p):
		"""
		计算闲家是否满足强攻和密
		"""
		# 闲家卡牌存在保证
		cards = [card % 100 for card in curr_p.combine_cards]
		duplicate_cards = set([card for card in cards if cards.count(card) > 1])
		if duplicate_cards:
			return [FarmerAction.SP, FarmerAction.QG, FarmerAction.MI]
		return [FarmerAction.SP]

	def calc_action_reward(self, lose_id, win_id, action):
		"""
		分别设置赢家和输家奖励
		"""
		init_reward = 0.2
		if action in [FarmerAction.QG, FarmerAction.FAN, LandLordAction.KAI]:
			action_reward = {
				FarmerAction.QG: init_reward + 0.1,
				FarmerAction.FAN: init_reward + 0.15,
				LandLordAction.KAI: init_reward + 0.2
			}[action]
		else:
			action_reward = init_reward
		self.players[win_id].action_reward = action_reward
		self.players[lose_id].action_reward = -action_reward

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
		# 计算玩家对象手牌好坏通过奖励表示
		for curr_p in self.players:
			rewards = self.split_big_and_small_by_cards(curr_p.curr_hand_cards)
			if self.winner_id == curr_p.player_id:
				curr_p.cards_reward += rewards
			else:
				curr_p.cards_reward -= rewards

	def split_big_and_small_by_cards(self, cards: list):
		"""
		拆分大小扑
		"""
		# 卡牌边界值
		bound = 5

		# 大扑与小扑
		big_cards = cards[:2]
		small_cards = cards[2:]

		# 计算卡牌好坏，是否能够组成水鱼或属水鱼天下
		big_has_pairs = self.calc_has_by_pairs(big_cards)
		small_has_pairs = self.calc_has_by_pairs(small_cards)

		# 1.处理设置存在癞子牌时，奖励设置
		# 判断水鱼天下
		if big_has_pairs == small_has_pairs:
			return 0.025 * bound

		# 判断水鱼
		elif big_has_pairs and small_has_pairs:
			return 0.015 * bound

		elif big_has_pairs:
			return 0.01 * bound + ((self.calc_split_attack_cards_by_sum(small_cards) - bound) * 0.025)

		# 豹子均不存在
		else:
			# 计算大扑与小扑值大小
			big_value = self.calc_split_attack_cards_by_sum(big_cards)
			small_value = self.calc_split_attack_cards_by_sum(small_cards)

			return (big_value - bound) * 0.025 + (small_value - bound) * 0.025

	@staticmethod
	def calc_has_by_pairs(cards: list):
		"""
		判断是否存在豹子[对子]
		"""
		return cards[0] if len(set(cards)) == 1 else False

	@staticmethod
	def calc_split_attack_cards_by_sum(cards: list):
		"""
		计算分扑卡牌值
		"""
		tmp_values = []
		for card in cards:
			# 判断花牌值，花牌的值为零
			if card % 100 > 9:
				tmp_values.append(0.5)
			# 当不是花牌时，则计算卡牌值大小
			else:
				tmp_values.append(card % 10)

		# 判断当前卡牌扑组合值是否大于10
		# 当前卡牌扑组合值最大只能为9
		if sum(tmp_values) < 10:
			return sum(tmp_values)
		else:
			# 11.5 % 10 -> 1.5 / 10 -> 0.15
			tmp_values = sum(tmp_values) % 10 / 10
			return tmp_values

	def compare_qg(self, curr_id, action):
		"""
		todo: 闲家首次选择操作为强攻时
			1.庄家只能选择开，比牌大小判定输赢家

		强攻分析:
			1.闲家强攻，庄家必开
			2.庄家两扑比闲家大，闲家赔2倍
			3.闲家两扑比庄家大，庄家赔2倍
		"""
		# 记录上一个动作和上一位玩家操作
		self.last_action = action
		self.last_player_id = curr_id

		# 闲、庄添加暗牌后的分扑，并更新为当前手牌
		farmer_cards, landlord_cards = self.calc_compare_cards()
		self.update_hand_cards(farmer_cards, landlord_cards)

		# todo: 判断本局游戏是否存在水鱼
		# 打开暗牌比较大小，判断是否存在水鱼或水鱼天下
		# 检测闲家(庄家)手牌是否为水鱼或水鱼天下
		self.check_sy_tx(farmer_cards, landlord_cards, curr_id)

		# todo: 存在水鱼时，赢家已产生，下述流程不再继续，不存在赢家时，则继续下述流程判断
		if self.winner_id is None:
			# 比较卡牌大小，根据输出结果来判定输赢家
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# todo: [闲家为赢家]
			# 闲家两扑都必庄家大，闲家赢
			if big_res == const.IS_MORE and small_res == const.IS_MORE:
				# 1.计算赢家奖励
				self.winner_id = curr_id
				lose_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(lose_id, self.winner_id, action)

				# [闲家大小扑都大]
				if big_res == const.IS_MORE and small_res == const.IS_MORE:
					self.calc_extra_reward(lose_id, self.winner_id, 0.2)

				# 2.计算本局卡牌值奖励
				self.calc_cards_by_reward()

			# todo: [庄家为赢家]
			else:
				# 1.计算赢家奖励
				self.winner_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(curr_id, self.winner_id, action)

				# [闲家大小扑都小]
				if big_res == const.IS_LESS and small_res == const.IS_LESS:
					self.calc_extra_reward(curr_id, self.winner_id, 0.2)

				# [闲家大小扑其中一扑小]
				if big_res == const.IS_LESS or small_res == const.IS_LESS:
					self.calc_extra_reward(curr_id, self.winner_id, 0.1)

				# 2.计算本局卡牌值奖励
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
			self.update_hand_cards(farmer_cards, landlord_cards)

			# todo: 判断本局游戏是否存在水鱼
			# 打开暗牌比较大小，判断是否存在水鱼或水鱼天下
			# 检测闲家(庄家)手牌是否为水鱼或水鱼天下
			self.check_sy_tx(farmer_cards, landlord_cards, curr_id)

			# todo: 存在水鱼时，赢家已产生，下述流程不再继续，不存在赢家时，则继续下述流程判断
			if self.winner_id is None:
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# todo: [闲家为赢家]
				if big_res == const.IS_MORE and small_res == const.IS_MORE:
					# 1.计算赢家奖励
					self.winner_id = self.select_next_id(curr_id)
					self.players[self.winner_id].is_winner = True
					self.calc_action_reward(curr_id, self.winner_id, action)

					# [闲家大小扑都大]
					if big_res == const.IS_MORE and small_res == const.IS_MORE:
						self.calc_extra_reward(curr_id, self.winner_id, 0.25)

					# 3.计算本局卡牌值奖励
					self.calc_cards_by_reward()

				# todo: [庄家为赢家]
				else:
					# 1.计算赢家奖励
					self.winner_id = curr_id
					lose_id = self.select_next_id(curr_id)
					self.players[self.winner_id].is_winner = True
					self.calc_action_reward(lose_id, self.winner_id, action)

					# [闲家大小扑都小]
					if big_res == const.IS_LESS and small_res == const.IS_LESS:
						self.calc_extra_reward(lose_id, self.winner_id, 0.25)

					# [闲家大小扑其中一扑小]
					if big_res == const.IS_LESS or small_res == const.IS_LESS:
						self.calc_extra_reward(lose_id, self.winner_id, 0.15)

					# 3.计算本局卡牌值奖励
					self.calc_cards_by_reward()

		# todo: [闲家为赢家] -> 不打开暗牌
		elif action == LandLordAction.REN:
			# 1.计算赢家奖励
			self.winner_id = self.select_next_id(curr_id)
			self.players[self.winner_id].is_winner = True
			self.calc_action_reward(curr_id, self.winner_id, action)

			# todo: 判断庄家认牌是否合理
			# 2.计算大小扑奖励
			farmer_cards, landlord_cards = self.calc_compare_cards()
			self.update_hand_cards(farmer_cards, landlord_cards)
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [闲家大小扑都大]
			# 庄家认牌合理，对庄家有利，庄家添加部分奖励
			if big_res == const.IS_MORE and small_res == const.IS_MORE:
				self.calc_extra_reward(self.winner_id, curr_id, 0.25)

			# [闲家大小扑其中一扑大]，庄家认牌不合理
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, self.winner_id, 0.15)

			# [闲家大小扑其中一扑小]，庄家认牌不合理
			if big_res == const.IS_LESS or small_res == const.IS_LESS:
				self.calc_extra_reward(curr_id, self.winner_id, 0.15)

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
		# todo: 闲撒 -> 庄杀 -> 闲信[不打开暗牌]
		if action == FarmerAction.XIN and self.last_action == LandLordAction.SHA:
			# todo: [庄家为赢家]
			# 1.计算赢家奖励
			self.winner_id = self.select_next_id(curr_id)
			self.players[self.winner_id].is_winner = True
			self.calc_action_reward(curr_id, self.winner_id, action)

			# 2.计算大小扑奖励
			farmer_cards, landlord_cards = self.calc_compare_cards()
			self.update_hand_cards(farmer_cards, landlord_cards)

			# 判断闲家认牌是否合理
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [闲家大小扑都小]
			if big_res == const.IS_LESS and small_res == const.IS_LESS:
				self.calc_extra_reward(self.winner_id, curr_id, 0.15)

			# [闲家大小扑其中一扑大]
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, self.winner_id, 0.1)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

		# todo: 闲撒 -> 庄杀 -> 闲反(比两扑)
		elif action == FarmerAction.FAN and self.last_action == LandLordAction.SHA:
			farmer_cards, landlord_cards = self.calc_compare_cards()
			self.update_hand_cards(farmer_cards, landlord_cards)

			# todo: 判断本局游戏是否存在水鱼
			# 打开暗牌比较大小，判断是否存在水鱼或水鱼天下
			# 检测闲家(庄家)手牌是否为水鱼或水鱼天下
			self.check_sy_tx(farmer_cards, landlord_cards, curr_id)

			# todo: 存在水鱼时，赢家已产生，下述流程不再继续，不存在赢家时，则继续下述流程判断
			if self.winner_id is None:
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# todo: 庄杀闲反
				# [庄杀闲反，闲家其中一扑大于庄家，则闲家为赢家]
				if big_res == const.IS_MORE or small_res == const.IS_MORE:
					# 1.计算赢家奖励
					self.winner_id = curr_id
					self.players[self.winner_id].is_winner = True
					lose_id = self.select_next_id(curr_id)
					self.calc_action_reward(lose_id, self.winner_id, action)

					# [闲家大小扑都大]
					if big_res == const.IS_MORE and small_res == const.IS_MORE:
						self.calc_extra_reward(lose_id, self.winner_id, 0.25)

					# [闲家大小扑其中一扑大]
					if big_res == const.IS_MORE or small_res == const.IS_MORE:
						self.calc_extra_reward(lose_id, self.winner_id, 0.15)

					# 3.计算本局卡牌值奖励
					self.calc_cards_by_reward()

				# [庄家为赢家]
				else:
					# 1.计算赢家奖励
					self.winner_id = self.select_next_id(curr_id)
					self.players[self.winner_id].is_winner = True
					self.calc_action_reward(curr_id, self.winner_id, action)

					# [闲家大小扑都大]
					if big_res == const.IS_LESS and small_res == const.IS_LESS:
						self.calc_extra_reward(curr_id, self.winner_id, 0.25)

					# 3.计算本局卡牌值奖励
					self.calc_cards_by_reward()

		# todo: 闲撒 -> 庄走 -> 闲信，无赢家[不开暗牌]
		elif action == FarmerAction.XIN and self.last_action == LandLordAction.ZOU:
			# 庄走闲信，[无赢家]
			self.winner_id = -1
			tmp_win_id = self.select_next_id(curr_id)
			farmer_cards, landlord_cards = self.calc_compare_cards()
			self.update_hand_cards(farmer_cards, landlord_cards)

			# 1.计算大小扑奖励
			big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
			# [闲家大小扑都大]
			if big_res == const.IS_LESS and small_res == const.IS_LESS:
				self.calc_extra_reward(tmp_win_id, curr_id, 0.15)

			# [闲家大小扑都小]
			if big_res == const.IS_MORE and small_res == const.IS_MORE:
				self.calc_extra_reward(curr_id, tmp_win_id, 0.25)

			# [闲家大小扑其中一扑大]
			if big_res == const.IS_MORE or small_res == const.IS_MORE:
				self.calc_extra_reward(tmp_win_id, curr_id, 0.2)

			# 3.计算本局卡牌值奖励
			self.calc_cards_by_reward()

		elif action == FarmerAction.FAN and self.last_action == LandLordAction.ZOU:
			# 庄走闲反: 比较牌型大小
			farmer_cards, landlord_cards = self.calc_compare_cards()
			self.update_hand_cards(farmer_cards, landlord_cards)

			# todo: 判断本局游戏是否存在水鱼
			# 打开暗牌比较大小，判断是否存在水鱼或水鱼天下
			# 检测闲家(庄家)手牌是否为水鱼或水鱼天下
			self.check_sy_tx(farmer_cards, landlord_cards, curr_id)

			# todo: 存在水鱼时，赢家已产生，下述流程不再继续，不存在赢家时，则继续下述流程判断
			if self.winner_id is None:
				big_res, small_res = Rules.compare_two_combine(farmer_cards, landlord_cards)
				# todo: 庄走闲反
				# [闲家为赢家]
				if big_res == const.IS_MORE and small_res == const.IS_MORE:
					# 1.计算赢家奖励
					self.winner_id = curr_id
					lose_id = self.select_next_id(curr_id)
					self.players[self.winner_id].is_winner = True
					self.calc_action_reward(lose_id, self.winner_id, action)

					# [闲家大小扑都大]
					if big_res == const.IS_MORE and small_res == const.IS_MORE:
						self.calc_extra_reward(lose_id, self.winner_id, 0.25)

					# 3.计算本局卡牌值奖励
					self.calc_cards_by_reward()

				# [庄家为赢家]
				else:
					# 1.计算赢家奖励
					self.winner_id = self.select_next_id(curr_id)
					self.players[self.winner_id].is_winner = True
					self.calc_action_reward(curr_id, self.winner_id, action)

					# [闲家大小扑都小]
					if big_res == const.IS_LESS and small_res == const.IS_LESS:
						self.calc_extra_reward(curr_id, self.winner_id, 0.2)

					# 3.计算本局卡牌值奖励
					self.calc_cards_by_reward()

		# 记录上一个动作
		self.last_action = action
		# 判断出赢家后，将不再更新上一位玩家ID
		if not self.winner_id:
			self.last_player_id = curr_id

	def check_sy_tx(self, farmer_cards, landlord_cards, curr_id):
		"""
		todo: is_bigger判断比较结果
			1.res == False: 庄家赢
			2.res == True: 闲家赢

		当前闲家出现出现水鱼或水鱼天下
		则直接进入比牌流程，输出赢家
		"""
		x_sy_tx = Rules.is_sy_tx(farmer_cards)  # 水鱼天下 [110, 110, 110, 110]
		x_sy = Rules.is_sy(farmer_cards)  # 水鱼 [110, 110, 119, 119]
		# [闲家为水鱼或水鱼天下]
		if x_sy_tx or x_sy:
			# [庄家为水鱼或水鱼天下]
			if self.calc_landlord_sy_tx(landlord_cards):
				# todo: 比较牌型大小[庄闲水鱼比牌]
				res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_Z_AND_X_SY)
				# [庄闲水鱼平局]
				if res == const.DRAW:
					self.winner_id = -1

				# [庄家水鱼赢]
				elif not res:
					# 1.计算赢家奖励
					if curr_id != 0:
						self.winner_id = self.select_next_id(curr_id)
						self.players[self.winner_id].is_winner = True
						self.calc_action_reward(curr_id, self.winner_id, 0.6)

					elif curr_id == 0:
						self.winner_id = curr_id
						lose_id = self.select_next_id(curr_id)
						self.players[self.winner_id].is_winner = True
						self.calc_action_reward(lose_id, self.winner_id, 0.6)

					# 2.计算卡牌值奖励
					self.calc_cards_by_reward()

				# [闲家水鱼赢]
				else:
					# 1.计算赢家奖励
					if curr_id == 1:
						self.winner_id = curr_id
						lose_id = self.select_next_id(curr_id)
						self.players[self.winner_id].is_winner = True
						self.calc_action_reward(lose_id, self.winner_id, 0.6)

					elif curr_id != 1:
						self.winner_id = self.select_next_id(curr_id)
						self.players[self.winner_id].is_winner = True
						self.calc_action_reward(curr_id, self.winner_id, 0.6)

					# 2.计算卡牌值奖励
					self.calc_cards_by_reward()

			# [闲家为水鱼或水鱼天下]
			else:
				# todo: 比较牌型大小
				res = Rules.is_bigger(farmer_cards, landlord_cards, const.COMPARE_X_SY)
				# [庄闲水鱼平局]
				if res == const.DRAW:
					self.winner_id = -1

				# [庄家赢]
				elif res:
					# 1.计算赢家奖励
					if curr_id != 0:
						self.winner_id = self.select_next_id(curr_id)
						self.players[self.winner_id].is_winner = True
						self.calc_action_reward(curr_id, self.winner_id, 0.6)

					elif curr_id == 0:
						self.winner_id = curr_id
						lose_id = self.select_next_id(curr_id)
						self.players[self.winner_id].is_winner = True
						self.calc_action_reward(lose_id, self.winner_id, 0.6)

					# 2.计算卡牌值奖励
					self.calc_cards_by_reward()

		# [庄家是否水鱼或水鱼天下]
		elif self.calc_landlord_sy_tx(landlord_cards):
			# 1.计算赢家奖励
			if curr_id != 0:
				self.winner_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(curr_id, self.winner_id, 0.4)

			elif curr_id == 0:
				self.winner_id = curr_id
				lose_id = self.select_next_id(curr_id)
				self.players[self.winner_id].is_winner = True
				self.calc_action_reward(lose_id, self.winner_id, 0.4)

			# 2.计算卡牌值奖励
			self.calc_cards_by_reward()

	def calc_compare_cards(self):
		"""
		统计当前卡牌值分布，用于计算输赢
		"""
		farmer_cards = []
		landlord_cards = []
		for player in self.players:
			# [闲家卡牌]
			if player.role == "farmer":
				get_combine_cards = player.combine_cards + player.dark_cards
				farmer_cards = self.get_over_combine_cards(get_combine_cards)

			# [庄家卡牌]
			elif player.role == "landlord":
				get_combine_cards = player.combine_cards + player.dark_cards
				landlord_cards = self.get_over_combine_cards(get_combine_cards)

		return farmer_cards, landlord_cards

	def update_hand_cards(self, farmer_cards, landlord_cards):
		"""
		更新最后分扑手牌
		"""
		for curr_p in self.players:
			if curr_p.role == "farmer":
				curr_p.update_cards(farmer_cards)

			elif curr_p.role == "landlord":
				curr_p.update_cards(landlord_cards)

	@staticmethod
	def get_over_combine_cards(get_combine_cards):
		"""
		分扑当前玩家手牌[大小扑]
		"""
		sort_big_small_cards = Rules.sort_big_small(get_combine_cards)
		return sort_big_small_cards

	@staticmethod
	def calc_landlord_sy_tx(cards):
		"""
		计算庄家水鱼或水鱼天下
		"""
		return True if Rules.is_sy_tx(cards) or Rules.is_sy(cards) else False