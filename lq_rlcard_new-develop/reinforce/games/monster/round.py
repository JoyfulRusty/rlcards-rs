# -*- coding: utf-8 -*-


from reinforce.const.monster.const import MAGIC_CARD, ActionType, BOGY, MONSTER, PUPIL, CARD_RANK

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

	def proceed_round(
			self,
			curr_p,
			players,
			action,
			golds,
			bust,
			state):
		"""
		判断操作流程
		"""
		self.dealer.action_history.append([action])
		self.traces.append((self.curr_player_id, action))

		# todo: 正常出牌
		if isinstance(action, int):
			self.record_picked_cards.append(action)
			if not self.has_bust(bust):
				next_player_id = (curr_p.player_id + 1) % len(players)
			else:
				next_player_id = self.select_next_id(curr_p.player_id, players)
		# todo: 捡牌
		else:
			# 判断捡牌后，是否破产
			curr_p.round_pick_cards = []
			curr_p.pick_cards(players, self.record_picked_cards, golds, bust, state)
			self.record_picked_cards = []
			# 破产后，移除持有卡牌并标记
			if curr_p.is_out:
				bust[self.update_bust(curr_p.player_id)] = True
				for card in curr_p.curr_hand_cards:
					if card in self.round_cards:
						self.round_cards.remove(card)
			# 破产时，迭代下一位操作玩家
			# 未破产，则捡牌玩家继续操作
			if curr_p.is_out:
				next_player_id = self.select_next_id(curr_p.player_id, players)
			else:
				next_player_id = curr_p.player_id
		self.last_player_id = curr_p.player_id
		return next_player_id

	@staticmethod
	def has_bust(bust):
		"""
		判断是否存在破产玩家
		"""
		return True if any(list(bust.values())) else False

	@staticmethod
	def update_bust(player_id):
		"""
		更新破产玩家
		"""
		return {
			0: "down",
			1: "right",
			2: "up",
			3: "left"
		}.get(player_id, 0)

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
			# todo: 查找第一位玩家到当前玩家之间的符合条件ID
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
		# todo: 查找第一位玩家到当前玩家之间符合条件的ID
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
		# todo: 下一位出牌玩家ID为当前玩家ID
		if next_player_id is None:
			next_player_id = curr_id

		return next_player_id

	def get_state(
			self,
			curr_p,
			dealer,
			other_hand_cards,
			hand_card_nums,
			bust,
			round_cards):
		"""
		状态数据
		"""
		state = {
			'actions': self.get_actions(curr_p),
			'traces': self.traces,
			'landlord_id': dealer.landlord_id,
			'self': curr_p.player_id,
			'curr_hand_cards': curr_p.curr_hand_cards,
			'other_hand_cards': other_hand_cards,
			'hand_card_nums': hand_card_nums,
			'played_cards': self.dealer.played_cards,
			'all_cards': round_cards,
			'action_history': self.dealer.action_history,
			'bust': bust
		}

		return state

	def get_actions(self, curr_p):
		"""
		获取合法动作
		"""
		if not curr_p.curr_hand_cards:
			return []

		# todo: 计算合法动作
		self.legal_actions = self.get_valid_actions(
			self.traces,
			curr_p,
			self.last_player_id
		)
		return self.legal_actions

	def get_valid_actions(self, traces, curr_p, last_p):
		"""
		获取合法出牌动作
		"""
		# 首次出牌只能出方块J
		if last_p is None:
			return self.dealer.jd_card

		# 上一个动作为捡牌时，则下一个动作将随机出牌
		if traces[-1][1] == ActionType.PICK_CARDS:
			return curr_p.curr_hand_cards

		return self.get_gt_cards(curr_p, traces)

	def get_gt_cards(self, curr_p, traces):
		"""
		todo: 计算能打出的牌
		"""
		gt_cards = []
		self.calc_can_picks(traces)
		gt_cards = self.get_can_play_cards(traces, curr_p.curr_hand_cards, gt_cards, self.can_pick)
		return [ActionType.PICK_CARDS] if not gt_cards else gt_cards

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
	if is_universal_card(card):
		return True
	if last_card == MONSTER:
		return True if card in BOGY else False
	if last_card in BOGY:
		if card in BOGY and compare_cards(card, last_card):
			return True
		return True if card in PUPIL else False
	if last_card in PUPIL:
		if card in PUPIL and value_tu_di(card, last_card):
			return True
		return True if card == MONSTER else False

def target_magic_card(last_card, traces):
	"""
	判断万能牌的使用
	"""
	if last_card == MAGIC_CARD and traces[-2][1] != ActionType.PICK_CARDS:
		if (traces[-2][1] % 100) in BOGY:
			return PUPIL[0]
		return MONSTER if (traces[-2][1]) % 100 in PUPIL else BOGY[0]
	return last_card

def compare_cards(card_1, card_2):
	"""
	比较卡牌大小
	"""
	key = []
	for card in [card_1, card_2]:
		key.append(CARD_RANK.index(card))
	return True if key[0] > key[1] else False

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
	return False if card != MAGIC_CARD else True