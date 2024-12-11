# -*- coding: utf-8 -*-

from random import choice
from collections import Counter

from rlcards.const.monster.const import *


class MonsterRound:
	"""
	打妖怪对局
	"""
	def __init__(self, np_random, round_cards, dealer):
		"""
		初始化属性参数
		"""
		self.dealer = dealer
		self.np_random = np_random
		self.round_cards = round_cards
		self.traces = []  									# 桌牌(记录/追踪玩家出牌)
		self.can_pick = False                               # 能否捡牌
		self.landlord_id = None                             # 持有方块J玩家ID
		self.last_player_id = None  						# 上一位玩家ID
		self.curr_player_id = None                          # 当前玩家ID
		self.legal_actions = []                             # 合法动作
		self.record_picked_cards = []  						# 记录玩家牌行为

	def proceed_round(self, curr_player, players, action, golds, bust, curr_state):
		"""
		判断操作流程
		"""
		pay_after_golds = 0.0
		self.dealer.action_history.append([action])
		self.traces.append((self.curr_player_id, action))

		# TODO: 记录出牌操作
		if isinstance(action, int):
			self.record_picked_cards.append(action)
			# 无破产玩家
			if not self.has_bust(bust):
				next_player_id = (curr_player.player_id + 1) % len(players)
			# 有破产玩家，破产玩家将不可选
			else:
				next_player_id = self.select_next_id(curr_player.player_id, players)

		# TODO: 捡牌
		else:
			# 捡牌之后金币数量(判断是否处于破产状态)
			# 玩家破产后出的牌将不再可见
			self.dealer.played_cards = {"down": [], 'right': [], 'up': [], 'left': []}
			pay_after_golds = curr_player.pick_cards(players, self.record_picked_cards, golds, bust, curr_state)
			self.record_picked_cards = []
			# 记录破产，并删除可选卡牌
			if curr_player.is_out:
				bust[self.update_bust(curr_player.player_id)] = True
				for card in curr_player.curr_hand_cards:
					if card in self.round_cards:
						self.round_cards.remove(card)
			# 计算是否存在破产玩家
			# TODO:
			#   1.玩家捡牌，计算完金币后，当前玩家没有破产，当前出牌玩家为自己
			#   2.玩家捡牌，计算完金币后，当前玩家破产，下一位玩家为出牌玩家并随机出牌
			#     1.当前玩家不是破产，如果下一位玩家为破产玩家，但破产玩家将不再继续游戏，理应为当前破产玩家的下一位玩家
			#     2.当前玩家是下一位玩家不是破产玩家

			# 破产玩家，将不可选
			if curr_player.is_out:
				next_player_id = self.select_next_id(curr_player.player_id, players)
			else:
				next_player_id = curr_player.player_id

		self.last_player_id = curr_player.player_id

		return pay_after_golds, next_player_id

	def has_bust(self, bust):
		"""
		判断是否存在破产玩家
		"""
		for pos, flag in bust.items():
			if not flag:
				continue
			return True
		return False

	def update_bust(self, player_id):
		"""
		更新破产玩家
		"""
		if player_id == 0:
			return 'down'
		elif player_id == 1:
			return 'right'
		elif player_id == 2:
			return 'up'
		return 'left'

	@staticmethod
	def select_next_id(curr_id, players):
		"""
		搜索下一个符合出牌的玩家
		"""
		# 下一位出牌玩家
		next_player_id = None
		if curr_id != 3:
			# 计算当前玩家ID到最后一位玩家的ID
			for next_id in range(curr_id + 1, len(players)):
				if players[next_id].is_out:
					continue
				if players[next_id].is_all:
					continue
				next_player_id = next_id
				break
			# 当下一位玩家ID为None时，则并未找到符合条件的玩家ID
			# TODO: 查找第一位玩家到当前玩家之间的符合条件ID
			if next_player_id is None:
				for next_id in range(0, curr_id):
					if next_id == 0 and players[next_id].is_out:
						continue
					if players[next_id].is_out:
						continue
					if players[next_id].is_all:
						continue
					next_player_id = next_id
					break
		# TODO: 查找第一位玩家到当前玩家之间符合条件的ID
		else:
			for next_id in range(0, curr_id):
				if players[next_id].is_out:
					continue
				if players[next_id].is_all:
					continue
				next_player_id = next_id
				break
		# 当都不存在，说明不存在符合下一位出牌的玩家
		# 其他玩家破产或当前手牌已经打完
		# TODO: 下一位出牌玩家ID为当前玩家ID
		if next_player_id is None:
			next_player_id = curr_id

		return next_player_id

	def get_state(self, curr_player, dealer, other_hand_cards, hand_card_nums, bust, round_cards):
		"""
		状态数据
		"""
		state = {
			'actions': self.get_actions(curr_player),
			'traces': self.traces,
			'landlord_id': dealer.landlord_id,
			'self': curr_player.player_id,
			'curr_hand_cards': curr_player.curr_hand_cards,
			'other_hand_cards': other_hand_cards,
			'hand_card_nums': hand_card_nums,
			'played_cards': self.dealer.played_cards,
			'all_cards': round_cards,
			'action_history': self.dealer.action_history,
			'bust': bust
		}

		return state

	def get_actions(self, curr_player):
		"""
		获取合法动作
		"""
		if not curr_player.curr_hand_cards:
			return []

		# 计算当前玩家合法动作
		self.legal_actions = self.get_valid_actions(
			self.traces,
			curr_player,
			self.last_player_id
		)

		return self.legal_actions

	def get_valid_actions(self, traces, curr_player, last_player=None):
		"""
		获取acton规则:
		1.第一次出牌，必须先出方块J(JD)，也就是第一个action必须为jd
		2.第二次获取需要根据上一位出牌来进行获取
			2.1 玩家要得起，根据游戏规则来进行action动作筛选(游戏牌值)
			2.2 玩家要得起，但是不要，进行捡牌(PICK_CARDS)
			2.3 玩家要不起，进行捡牌
		"""
		# 首次出牌只能出方块J
		if last_player is None:
			actions = self.dealer.jd_card

		# 上一个动作为捡牌时，则下一个动作将随机出牌
		elif traces[-1][1] == ActionType.PICK_CARDS:
			all_actions = curr_player.curr_hand_cards
			actions = [choice(list(all_actions))]

		# 计算大于上一位玩家的手牌
		else:
			actions = self.get_gt_cards(curr_player, traces)

		return actions

	def get_gt_cards(self, player, traces):
		"""
		TODO: 计算能打出的牌
		"""
		gt_cards = []
		# 判断是否符合首次捡牌规则
		self.calc_can_picks(traces)
		gt_cards = self.get_can_play_cards(traces, player.curr_hand_cards, gt_cards, self.can_pick)
		# 无合法动作时，仅能捡牌
		if not gt_cards:
			return [ActionType.PICK_CARDS]
		return gt_cards

	@staticmethod
	def get_can_play_cards(traces, hand_cards, gt_cards, can_pick):
		"""
		获取玩家能出的卡牌
		"""
		if len(hand_cards) != 0 and len(traces) != 0:
			last_card = traces[-1][1] % 100
			if last_card == MAGIC_CARD:
				last_card = target_magic_card(last_card, traces)
			for card in hand_cards:
				card_value = card % 100
				if attack_valid(card_value, last_card):
					gt_cards.append(card)
			if can_pick:
				gt_cards.append(ActionType.PICK_CARDS)
		return gt_cards

	def calc_can_picks(self, traces):
		"""
		判断首次捡牌是否符合规则
		"""
		tu_di_cards = [3, 8, 5]
		pick_flag = [10, 20]
		for i, trace in enumerate(traces):
			# 无师傅牌，则判断万能牌是否作为师傅牌
			if traces[i][1] == ActionType.PICK_CARDS:
				continue
			# 存在师傅牌则直接跳出
			if (traces[i][1] % 100) == pick_flag[0]:
				self.can_pick = True
				break
			# 未出现师傅牌时，判断万能牌是否作为师傅牌打出
			# 前一张卡牌动作不能捡牌，否则不满足查找条件
			if traces[i - 1][1] != ActionType.PICK_CARDS:
				if (traces[i - 1][1] % 100) in tu_di_cards and (traces[i][1] % 100) in pick_flag:
					self.can_pick = True
					break

def attack_valid(card, last_card):
	"""
	获取玩家可出的牌
	"""
	if not card or not last_card:
		return False

	# 师傅牌
	if last_card == MONSTER:
		if card in BOGY:
			return True
		return False

	# 万能牌
	if is_universal_card(card):
		return True

	# 上一张为妖怪牌
	if last_card in BOGY:
		if card in PUPIL:
			return True
		if card in BOGY and compare_cards(card, last_card):
			return True
		return False

	# 上一张为徒弟牌
	if last_card in PUPIL:
		if card == MONSTER:
			return True
		if card in PUPIL and value_tu_di(card, last_card):
			return True
		return False

def target_magic_card(last_card, traces):
	"""
	判断万能牌的使用
	"""
	if last_card == MAGIC_CARD and traces[-2][1] != ActionType.PICK_CARDS:
		if (traces[-2][1] % 100) in BOGY:
			last_card = PUPIL[0]
		elif (traces[-2][1]) % 100 in PUPIL:
			last_card = MONSTER
		else:
			last_card = BOGY[0]
	return last_card

def compare_cards(card_1, card_2):
	"""
	比较卡牌大小
	"""
	key = []
	for card in [card_1, card_2]:
		key.append(CARD_RANK.index(card))
	if key[0] > key[1]:
		return True
	return False

def value_tu_di(card1, card2):
	"""
	在两张牌都是徒弟的情况下，比较大小(徒弟牌: 5>8>3)
	"""
	if card1 == PUPIL[-1] and card2 != PUPIL[-1]:
		return True
	if card1 == PUPIL[1] and card2 == PUPIL[0]:
		return True
	return False

def is_universal_card(card):
	"""
	判断是否为万能牌
	"""
	return card == MAGIC_CARD