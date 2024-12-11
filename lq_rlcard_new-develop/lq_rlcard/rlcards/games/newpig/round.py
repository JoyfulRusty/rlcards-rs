# -*- coding: utf-8 -*-

from rlcards.const.pig import const
from rlcards.games.newpig.move import MovesGenerator
from rlcards.games.newpig.utils import get_move_type

class GzRound:
	"""
	供猪流程
	"""
	def __init__(self, np_random, dealer, players):
		"""
		初始化参数
		"""
		self.random = np_random
		self.dealer = dealer
		self.players = players
		self.winner_id = None
		self.curr_player_id = None
		self.big_slam_position = None
		self.all_hong_position = None
		self.collect_score_role = None
		self.turn_cards = []
		self.record_round_cards = []
		self.has_collect = False
		self.all_cards = const.ALL_POKER[:]
		self.round_cards = {role: [] for role in const.ALL_ROLE}
		self.round_scores = {role: [] for role in const.ALL_ROLE}
		self.receive_score_cards = {role: [] for role in const.ALL_ROLE}
		self.remain_score_cards = {role: [] for role in const.ALL_ROLE}
		self.all_receive_cards = {role: [] for role in const.ALL_ROLE}

	def proceed_round(self, action):
		"""
		todo: 计算本轮收分流程
		"""
		self.has_collect = False
		self.record_round_cards.append(action)
		# 从所有牌盒中删除当前出牌
		if action in self.all_cards:
			self.all_cards.remove(action)
		# 判断是否满足收分条件[必须为一轮[4次]]
		if all(self.round_cards.values()):
			# 开始计算收分
			self.calc_collect_by_position()
			self.calc_collect_scores_by_round()
			# 重新记录本轮记录数据
			self.round_cards = {role: [] for role in const.ALL_ROLE}
			self.record_round_cards = []
			if self.winner_id is None:
				self.curr_player_id = self.update_curr_id()
			self.has_collect = True

	def update_curr_id(self):
		"""
		todo: 谁收分谁先出牌
		收分后，更新当前玩家ID
		"""
		return {
			"down": 0,
			"right": 1,
			"up": 2,
			"left": 3
		}[self.collect_score_role]

	def get_legal_actions(self, player):
		"""
		获取合法动作
		"""
		mg = MovesGenerator(player)
		turn_cards = self.turn_cards
		rival_move = []
		if len(turn_cards) != 0:
			rival_move = turn_cards[0:1]  # 优先跟上一家同一种花色
		rival_type = get_move_type(rival_move)
		rival_move_type = rival_type['type']
		rival_suit = rival_type.get('suit', None)
		moves = list()
		if rival_move_type == const.TYPE_0_PASS:
			moves = mg.gen_moves()
		elif rival_move_type == const.TYPE_1_SINGLE:
			moves = mg.gen_can_play_cards(rival_suit)
		return moves

	def find_same_suit_cards_by_round(self):
		"""
		计算一轮中相同花色的卡牌
		"""
		same_suit_cards = []
		if not self.record_round_cards:
			return []
		first_card = self.record_round_cards[0]
		suit = first_card // 100
		for rc in self.record_round_cards:
			if rc // 100 == suit:
				same_suit_cards.append(rc)
		return same_suit_cards

	def get_max_card_by_round(self):
		"""
		计算一轮中最大分数
		"""
		same_suit_cards = self.find_same_suit_cards_by_round()
		if not same_suit_cards:
			return []
		return [max(same_suit_cards)]

	def calc_collect_by_position(self):
		"""
		计算一轮中分数花色最大的玩家
		"""
		self.collect_score_role = sorted(self.round_cards.items(), key=lambda x: x[1])[-1][0]

		# 谁收分，谁接着出牌
		collect_p = self.select_player_by_position(self.collect_score_role)
		self.all_receive_cards[collect_p.role].extend(sum(self.round_cards.values() or [], []))
		self.curr_player_id = collect_p.player_id

	def calc_receive_score_cards(self):
		"""
		计算剩余的分数卡牌
		"""
		# 重置当前的分数卡牌
		self.receive_score_cards = {role: [] for role in const.ALL_ROLE}
		for curr_p in self.players:
			self.receive_score_cards[curr_p.role].extend(curr_p.receive_score_cards or [])
		return self.receive_score_cards

	def calc_remain_score_cards(self):
		"""
		计算当前每位玩家手中的分数卡牌
		"""
		self.remain_score_cards = {role: [] for role in const.ALL_ROLE}
		for curr_p in self.players:
			self.remain_score_cards[curr_p.role].extend(curr_p.curr_score_cards or [])
		return self.remain_score_cards

	def check_big_slam(self):
		"""
		检查大满贯玩家
		"""
		for key, value in self.receive_score_cards.items():
			if len(value) == const.BIG_SLAM_NUM:
				return value
		return False

	def check_all_hong(self):
		"""
		检查全红
		"""
		for key, value in self.receive_score_cards.items():
			if len(value) == len(const.ALL_RED):
				return value
		return False

	def calc_big_slam_scores(self):
		"""
		todo: 条件满足计算大满贯分数
		"""
		scores = 0
		slam_res = self.check_big_slam()
		curr_p = self.select_player_by_position(slam_res)
		for sc in self.receive_score_cards[slam_res]:
			s = const.SCORE_CARDS.get(sc, 0)
			scores += abs(s)
		scores -= const.SCORE_CARDS.get(const.BAN_CARD)
		scores *= 2
		curr_p.curr_scores = scores
		self.big_slam_position = slam_res

	def calc_all_hong_scores(self, has_ban=False):
		"""
		todo: 条件满足才会计算全红分数
		"""
		scores = 0
		hong_res = self.check_all_hong()
		curr_p = self.select_player_by_position(hong_res)
		for sc in self.receive_score_cards[hong_res]:
			s = const.SCORE_CARDS.get(sc, 0)
			scores += abs(s)
		if has_ban:
			scores -= const.SCORE_CARDS.get(const.BAN_CARD)
			scores *= 2
		curr_p.curr_scores = scores
		self.all_hong_position = hong_res

	def select_player_by_position(self, position):
		"""
		根据位置查找当前玩家
		"""
		if position == 'down':
			return self.players[0]
		elif position == 'right':
			return self.players[1]
		elif position == 'up':
			return self.players[2]
		elif position == 'left':
			return self.players[3]

	def check_score_cards_by_round(self):
		"""
		todo: 检查一轮结束后，分牌是否出完
		检查一轮结束后收分时调用，分牌出完，则游戏结束
		一轮结束后，检查分数牌
		"""
		remain_score_cards = sum(self.remain_score_cards.values() or [], [])
		if remain_score_cards:
			return False
		# 无剩余分牌时，则计算当前对局分数最大玩家，游戏结束
		self.winner_id = self.calc_winner()

	def calc_winner(self):
		"""
		计算分数最多的玩家
		"""
		scores = {'down': 0, 'right': 0, 'up': 0, 'left': 0}
		for curr_p in self.players:
			scores[curr_p.role] += curr_p.curr_scores
		self.round_scores = sorted(scores.items(), key=lambda x: x[1])
		return self.round_scores[-1][0]

	def calc_collect_scores_by_round(self):
		"""
		计算一轮结束后的分牌及收取分牌玩家
		"""
		score_cards = []
		# 计算分牌
		for pos, card in self.round_cards.items():
			if not card:
				continue
			if card[0] in const.SCORE_CARDS:
				score_cards.extend(card)
		if score_cards:
			collect_p = self.select_player_by_position(self.collect_score_role)
			collect_p.receive_score_cards.extend(score_cards)
			self.receive_score_cards[self.collect_score_role].extend(score_cards)
			has_ban = const.BAN_CARD in self.receive_score_cards[self.collect_score_role]
			if self.check_big_slam():
				self.calc_big_slam_scores()
			elif self.check_all_hong():
				self.calc_all_hong_scores(has_ban)
			elif has_ban:
				if len(self.receive_score_cards[self.collect_score_role]) == 1:
					total_score = const.SCORE_CARDS.get(const.BAN_CARD, 0)
				else:
					# 存在变压器时，则重新计算分数
					total_score = 0
					for sc in self.receive_score_cards[self.collect_score_role]:
						total_score += const.SCORE_CARDS.get(sc, 0)
					total_score -= const.SCORE_CARDS.get(const.BAN_CARD, 0)
					total_score *= 2
				collect_p = self.select_player_by_position(self.collect_score_role)
				collect_p.curr_scores += total_score
			else:
				add_scores = 0
				for sc in score_cards:
					add_scores += const.SCORE_CARDS.get(sc, 0)
				collect_p = self.select_player_by_position(self.collect_score_role)
				collect_p.curr_scores += add_scores

			self.check_score_cards_by_round()

	def get_state(self, curr_p, last_action, other_hand_cards):
		"""
		获取玩家状态数据
		------------------------------
		1.上一位玩家位置
		2.本轮收牌玩家位置
		3.上一个动作
		4.本轮收牌最大动作
		5.玩家合法动作
		6.对局中已经亮的牌
		7.本轮中已经亮的牌
		8.对局中出现过的分牌
		9.本轮中出现的分牌
		10.剩余的分牌
		11.每一位玩家手中的分牌
		12.玩家当前手牌
		13.其他玩家手牌
		14.其他玩家打出的手牌
		15.剩余卡牌
		16.动作序列
		17.一轮的动作序列
		------------------------------
		"""
		state = {
			'self': curr_p.player_id,
			'last_action': self.calc_last_action(last_action),
			'collect_position': self.collect_score_role,
			'legal_actions': self.calc_legal_actions(curr_p),
			'max_round_action': self.get_max_card_by_round(),
			'turn_light_cards': self.get_turn_light_cards(),
			'round_light_cards': self.get_round_light_cards(),
			'turn_score_cards': self.get_turn_score_cards(),
			'round_score_cards': self.get_round_score_cards(),
			'turn_cards': self.turn_cards,
			'round_cards': self.get_round_cards(),
			'curr_hand_cards': curr_p.curr_hand_cards,
			'other_hand_cards': other_hand_cards,
			'played_cards': self.dealer.played_cards,
			'light_cards': self.get_player_light_cards(),
			'all_cards': self.all_cards,
			'all_receive_cards': self.all_receive_cards,
			'remain_score_cards': self.calc_remain_score_cards(),
			'receive_score_cards': self.calc_receive_score_cards(),
			'round_same_suit_action': self.calc_same_suit_cards(),
			'round_other_suit_action': self.calc_other_suit_cards(),
		}

		return state

	def calc_same_suit_cards(self):
		"""
		计算相同类型卡牌
		"""
		same_suit_cards = self.find_same_suit_cards_by_round()
		if not same_suit_cards:
			return []
		return same_suit_cards

	def calc_other_suit_cards(self):
		"""
		计算与首家出牌不同花色的卡牌
		"""
		same_suit_cards = self.calc_same_suit_cards()
		diff_cards = list(set(self.record_round_cards) ^ set(same_suit_cards))
		return diff_cards

	def calc_legal_actions(self, curr_p):
		"""
		计算合法动作
		"""
		legal_actions = self.get_legal_actions(curr_p)
		if not legal_actions:
			return [curr_p.played_cards[-1]]
		# 判断当前羊牌是否能出
		return legal_actions

	def can_play_yang_card(self):
		"""
		判断能否出羊牌
		"""
		count = 0
		for card in self.turn_cards:
			if card // 100 == 1:
				count += 1
				continue
			if count > 3:
				return True
		return False

	def calc_last_action(self, last_action):
		"""
		计算上一个动作
		"""
		if not self.all_cards:
			return [self.turn_cards[-2]]
		return [last_action] if last_action else last_action

	def get_round_cards(self):
		"""
		获取一轮中的卡牌
		"""
		return self.record_round_cards

	def get_turn_light_cards(self):
		"""
		计算对局中已经出现的亮牌
		"""
		if not self.turn_cards:
			return []
		light_cards = []
		for card in self.turn_cards:
			if card in const.CAN_LIGHT_CARDS:
				light_cards.append(card)
		return light_cards

	def get_round_light_cards(self):
		"""
		计算一轮中出现的亮牌
		"""
		if not self.record_round_cards:
			return []
		light_cards = []
		for card in self.record_round_cards:
			if card in const.CAN_LIGHT_CARDS:
				light_cards.append(card)
		return light_cards

	def get_turn_score_cards(self):
		"""
		计算对局中已经出现过的分牌
		"""
		if not self.turn_cards:
			return []
		score_cards = []
		for card in self.turn_cards:
			if card in const.SCORE_CARDS:
				score_cards.append(card)
		return score_cards

	def get_round_score_cards(self):
		"""
		计算一轮中出现的分牌
		"""
		if not self.record_round_cards:
			return []
		score_cards = []
		for card in self.record_round_cards:
			if card in const.SCORE_CARDS:
				score_cards.append(card)
		return score_cards

	def get_player_light_cards(self):
		"""
		获取玩家亮牌
		"""
		light_cards = {role: [] for role in const.ALL_ROLE}
		for curr_p in self.players:
			light_cards[curr_p.role].extend(curr_p.light_cards)
		return light_cards