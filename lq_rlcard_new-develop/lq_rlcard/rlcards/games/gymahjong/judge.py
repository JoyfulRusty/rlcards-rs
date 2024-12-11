# -*- coding: utf-8 -*-

import numpy as np

from copy import deepcopy
from collections import Counter

from rlcards.const.mahjong import const
from rlcards.utils.mahjong.xxc import RuleXXC
from rlcards.const.mahjong.const import ActionType, HuPaiType


class MahjongJudge:
	"""
	麻将判断流程
	"""
	def __init__(self, np_random):
		"""
		TODO: 初始化参数
		"""
		self.np_random = np_random

	@staticmethod
	def do_pong(curr_player, last_card):
		"""
		TODO: 玩家选择碰，删除碰牌
		"""
		curr_player.remove_cards(last_card, 2)
		pile_cards = [ActionType.PONG, [last_card] * 3, curr_player.player_id]  # [动作类型， 卡牌， 玩家]
		curr_player.piles.append(pile_cards)

		# 清空操作
		curr_player.clear_operates()

	@staticmethod
	def do_ming_gang(curr_player, last_card):
		"""
		TODO: 玩家选择明杠，删除杠牌
		"""
		curr_player.remove_cards(last_card, 3)
		pile_cards = [ActionType.MING_GONG, [last_card] * 4, curr_player.player_id]  # [动作类型， 卡牌， 玩家]
		curr_player.piles.append(pile_cards)

		# 清空操作
		curr_player.clear_operates()

	@staticmethod
	def do_suo_gang(curr_player, dealer):
		"""
		TODO: 玩家选择索杠，删除索杠牌(还需删除之前的碰牌)
		"""
		# 判断当前摸牌是否在之前的碰牌中
		can_suo_gang = curr_player.calc_suo_gang(dealer.curr_mo_card)

		# 索杠，删除当前的摸牌
		if can_suo_gang:
			# 索杠仅删除当前摸牌
			curr_player.remove_cards(dealer.curr_mo_card, 1)
			pile_cards = [ActionType.SUO_GONG, [dealer.curr_mo_card] * 4, curr_player.player_id]  # [动作类型， 卡牌， 玩家]
			curr_player.piles.append(pile_cards)

			# 清空操作
			curr_player.clear_operates()

	@staticmethod
	def do_an_gang(curr_player, dealer):
		"""
		TODO: 玩家选择暗杠，删除暗杠牌
			1.起手牌四张，憨包杠
			2.摸牌凑四张
		"""
		mo_card = dealer.curr_mo_card
		# 计算起手牌后，是否存在憨包杠(未出过牌，且当前卡牌数量为14)
		cards_dict = Counter(curr_player.curr_hand_cards)
		for card, nums in cards_dict.items():
			if nums == 4 and card != mo_card:
				mo_card = card
				# 统计玩家憨包杠数量
				curr_player.han_bao_gang_count += 1

		# 删除玩家(暗)杠牌
		curr_player.remove_cards(mo_card, 4)
		pile_cards = [ActionType.AN_GONG, [mo_card] * 4, curr_player.player_id]  # [动作类型， 卡牌， 玩家]
		curr_player.piles.append(pile_cards)

		# 清空操作
		curr_player.clear_operates()

	@staticmethod
	def do_tian_ting(curr_player):
		"""
		TODO: 玩家选择报听
		"""
		# 听牌打出后，玩家叫牌
		tmp_cards = curr_player.curr_hand_cards[:]
		is_ting, ting_cards = RuleXXC.get_tian_ting_cards(curr_player.piles, tmp_cards)
		curr_player.ting_list = ting_cards

		# 删除听牌，并将手牌锁住
		for card in tmp_cards:
			if card in ting_cards:
				tmp_cards.remove(card)

		# 天听锁牌
		curr_player.lock_cards = tmp_cards[:]

		# 清空操作
		curr_player.clear_operates()

	@staticmethod
	def mi_hu(curr_player, players, dealer, last_player_id):
		"""
		TODO: 玩家选择闷胡
			1.锁住不包含当前摸牌的手牌
			2.记录闷胡信息
		"""
		curr_mo_card = dealer.curr_mo_card
		landlord_id = dealer.landlord_id
		tmp_cards = curr_player.curr_hand_cards[:]

		# 密胡锁牌🔒
		if not curr_player.lock_cards:
			for tmp_card in tmp_cards:
				if tmp_card == curr_mo_card:
					tmp_cards.remove(curr_mo_card)
					break
			curr_player.lock_cards = tmp_cards[:]
		else:
			tmp_cards.remove(curr_mo_card)
			curr_player.lock_cards = tmp_cards[:]

		# 计算密胡类型
		hu_type, hu_path = RuleXXC.get_hu_type(curr_player, players, last_player_id, landlord_id, curr_mo_card, True)
		if hu_type:
			# 闷胡信息[当前玩家，当前玩家摸牌，闷胡，胡牌类型]
			men_infos = {
				"player_id": curr_player.player_id,
				"men_card": dealer.curr_mo_card,
				"action": ActionType.MI_HU,
				"hu_type": hu_type
			}

			# 计算胡牌类型奖励，将其添加到动作奖励中
			lv = 0.01
			tmp_rewards = 0.0
			for hu in hu_type:
				scores = const.HU_PAI_SCORES[hu]
				curr_player.rewards += (lv * scores)
				tmp_rewards += round((lv * scores), 2)

			# 密胡时，其他两位玩家奖励
			for player in players:
				if player == curr_player:
					continue
				player.rewards -= round((tmp_rewards/2), 2)

			curr_player.men_card_infos.append(men_infos)

	@staticmethod
	def pick_hu(curr_player, players, dealer, last_player_id):
		"""
		TODO: 玩家选择捡胡
		"""
		curr_chu_card = dealer.table[-1]
		landlord_id = dealer.landlord_id
		# 捡胡锁牌🔒
		curr_player.lock_cards = curr_player.curr_hand_cards[:]

		# 计算捡胡类型
		hu_type, hu_path = RuleXXC.get_hu_type(curr_player, players, last_player_id, landlord_id, curr_chu_card)
		if hu_type:
			# 捡胡信息[当前玩家，上一位玩家出牌，捡胡，胡牌类型]
			pick_infos = {
				"player_id": curr_player.player_id,
				"pick_card": curr_chu_card,
				"action": ActionType.PICK_HU,
				"hu_type": hu_type
			}

			# 计算胡牌类型奖励
			lv = 0.01
			for hu in hu_type:
				scores = const.HU_PAI_SCORES[hu]
				# 计算当前胡牌与被胡牌奖励
				curr_player.rewards += round((lv * scores), 2)
				players[last_player_id].rewards -= round((lv * scores), 2)

			curr_player.pick_card_infos.append(pick_infos)

	@staticmethod
	def calc_operates_after_mo_pai(player, players, dealer, last_player_id):
		"""
		TODO: 计算当前玩家摸牌之后所具有的有效操作
		"""
		curr_mo_card = dealer.curr_mo_card
		landlord_id = dealer.landlord_id

		# 摸牌后有效动作
		valid_actions = []
		if not player:
			return valid_actions

		# 判断玩家是否胡得起
		hu_type, hu_path = RuleXXC.get_hu_type(player, players, last_player_id, landlord_id, curr_mo_card, True)
		if hu_type:
			valid_actions.append(ActionType.KAI_HU)
			valid_actions.append(ActionType.MI_HU)
			# 必胡条件，小于必胡数量时，则删除闷胡，直接执行胡开
			if dealer.left_count < const.XUE_LIU_LEFT_BI_HU:
				valid_actions.remove(ActionType.MI_HU)
			else:
				# 未进行过胡牌，也没闷胡过，则删除胡开
				if not player.hu_infos and player.bi_men_yi_shou:
					valid_actions.remove(ActionType.KAI_HU)

		if hu_type and len(player.gang_hou_chu_card) > 0:
			return valid_actions
		# 玩家补牌之后不能胡，则判断玩家其他操作(碰、杠、听)
		else:
			# 非庄家只能计算天听
			if not player.all_chu_cards and player.player_id != landlord_id:
				can_tian_ting = player.can_select_tian_ting_14()
				if can_tian_ting:
					# 玩家可以进行天听
					player.can_tian_ting = True
					valid_actions.append(ActionType.TIAN_TING)

			# 玩家处于天听或闷牌状态
			if player.tian_ting or len(player.hu_infos) > 0:
				# TODO: 计算玩家是否能够索杠得起
				is_suo_gong, gang_card = player.suo_gang_de_qi(curr_mo_card)
				# 当前摸牌是否能索杠
				if is_suo_gong and gang_card == curr_mo_card:
					valid_actions.append(ActionType.SUO_GONG)

				# TODO: 计算玩家是否能暗杠
				is_an_gong, gang_card = player.an_gang_de_qi(curr_mo_card)
				# 玩家能够暗杠时
				if is_an_gong:
					# 先计算杠之前的听牌，再计算杠之后的听牌
					tmp_cards = deepcopy(player.curr_hand_cards)
					tmp_cards.remove(gang_card)
					tmp_cards.remove(gang_card)
					tmp_cards.remove(gang_card)
					tmp_cards.remove(gang_card)

					# 比较杠牌前后的听牌
					ting_list1 = RuleXXC.get_ting_hu_list(tmp_cards)
					if ting_list1 == player.ting_list:
						valid_actions.append(ActionType.AN_GONG)

			# 不是天听和闷牌
			else:
				# 判断是否能索杠
				is_suo_gong, gang_card = player.suo_gang_de_qi(curr_mo_card)
				if is_suo_gong:
					valid_actions.append(ActionType.SUO_GONG)
				# 判断是否能暗杠
				is_an_gong, gang_card = player.an_gang_de_qi(curr_mo_card)
				if is_an_gong:
					valid_actions.append(ActionType.AN_GONG)

		return valid_actions

	@staticmethod
	def calc_operates_after_chu_pai(player, players, dealer, last_player_id):
		"""
		TODO: 计算上一位玩家出牌后，其他玩家所具有的有效操作
		"""
		curr_chu_card = dealer.table[-1]
		landlord_oid = dealer.landlord_id

		# 有效操作
		valid_actions = []  # 其他玩家出牌时，能碰，但可选择不碰(PASS)
		if not player.curr_hand_cards:
			return valid_actions

		# 天听或闷捡，则不能换牌
		if player.tian_ting or len(player.hu_infos) > 0:
			# 判断是否明杠得起
			is_ming_gong, gang_card = player.ming_gang_de_qi(curr_chu_card)
			if is_ming_gong:
				tmp_cards = deepcopy(player.curr_hand_cards)
				tmp_cards.remove(gang_card)
				tmp_cards.remove(gang_card)
				tmp_cards.remove(gang_card)

				# 判断是否相等，相等则可以进行明杠
				ting_list1 = RuleXXC.get_ting_hu_list(tmp_cards)
				if ting_list1 == player.ting_list:
					valid_actions.append(ActionType.MING_GONG)
		else:
			# 碰得起
			is_pong, pong_card = player.pong_de_qi(curr_chu_card)
			if is_pong:
				valid_actions.append(ActionType.PONG)

			# 明杠得起
			is_ming_gong, gang_card = player.ming_gang_de_qi(curr_chu_card)
			if is_ming_gong:
				valid_actions.append(ActionType.MING_GONG)

		# 判断是否能胡
		hu_type, hu_path = RuleXXC.get_hu_type(player, players, last_player_id, landlord_oid, curr_chu_card)
		if hu_type:
			valid_actions.append(ActionType.KAI_HU)  # 胡开
			valid_actions.append(ActionType.PICK_HU)  # 捡胡

			# 小于最小捡牌数，则不能再进行捡牌
			if dealer.left_count < const.MAX_PICK_NUMS:
				valid_actions.remove(ActionType.PICK_HU)
			else:
				# 胡牌类型为平胡时，则判断胡开条件(检查通行证)
				if hu_type == HuPaiType.PING_HU:
					if not player.hu_infos and not player.check_permit() and not player.tian_ting:
						valid_actions.remove(ActionType.KAI_HU)
				# 未进行过胡牌，也一次闷牌也没有，则不能胡开
				if not player.hu_infos and player.bi_men_yi_shou:
					valid_actions.remove(ActionType.KAI_HU)

		if valid_actions and dealer.left_count > const.MAX_PICK_NUMS:
			valid_actions.append(ActionType.G_PASS)

		return valid_actions

	def judge_name(self, game):
		"""
		TODO: 计算奖励
		"""
		payoffs = np.array(const.INIT_REWARD, dtype=np.float32)

		# 先计算[天听，杠，胡]奖励
		for player in game.players:
			payoffs[player.player_id] += round(player.rewards, 2)

		# 计算鸡牌分数奖励
		# self.calc_ji_cards_rewards(game, payoffs)
		self.calc_ji_cards_rewards_fk(game, payoffs)

		# 游戏结束流局，则返回当局玩家动作奖励及胡牌类型奖励
		if game.winner_id == -1:
			return payoffs
		else:
			# 游戏结束胡开，则添加胡开奖励，并返回结果
			payoffs[game.winner_id] += round(game.dynamics_reward, 2)

			# 胡开后，计算其他两位玩家奖励
			# 输家减去一半的奖励
			for player in game.players:
				if player.player_id != game.winner_id:
					lost_rewards = round(game.dynamics_reward, 2)
					payoffs[player.player_id] -= (lost_rewards / 2)  # 非胡牌玩家，减去胡牌玩家奖励的1/2

			return payoffs

	def judge_payoffs(self, game):
		"""
		TODO: 计算每一位玩家对应的奖励情况
		"""
		# 计算对局奖励
		payoffs = self.judge_name(game)

		# print("输出对局结束后的奖励: ", payoffs)

		return payoffs

	@staticmethod
	def calc_ji_cards_rewards(game, payoffs):
		"""
		计算鸡分奖励
		"""
		ji_rewards = 0.005
		result = game.round_over_xl_hz()
		# print("输出结果: ", result)
		for p_id, infos in result.items():
			payoffs[p_id] += round(infos['score'] * ji_rewards, 2)

	@ staticmethod
	def calc_ji_cards_rewards_fk(game, payoffs):
		"""
		计算房卡鸡牌奖励
		"""
		ji_rewards = 0.005
		result = game.round_over_fk(game.players, game.curr_action)
		# print("输出房卡麻将鸡牌分数结果: ", result)
		for p_id, infos in result.items():
			payoffs[p_id] += round(infos['total_score'] * ji_rewards, 2)