# -*- coding: utf-8 -*-

import numpy as np

from rlcards.const.pig import const
from rlcards.games.newpig.move import MovesGenerator
from rlcards.games.newpig.judge import GzJudge as Judge
from rlcards.games.newpig.round import GzRound as Round
from rlcards.games.newpig.dealer import GzDealer as Dealer
from rlcards.games.newpig.player import GzPlayer as Player
from rlcards.games.newpig.utils import lp_policy, get_move_type


class GzGame:
	"""
	供猪游戏流程
	"""
	def __init__(self, allow_step_back=False):
		"""
		初始化游戏流程
		"""
		self.judge = None
		self.round = None
		self.dealer = None
		self.players = None
		self.step_count = 0
		self.curr_state = {}
		self.last_action = None
		self.curr_player_id = None
		self.num_players = const.PLAYER_NUMS
		self.allow_step_back = allow_step_back
		self.np_random = np.random.RandomState()

		self.light_cards = {role: [] for role in const.ALL_ROLE}

	def init_game(self):
		"""
		初始化游戏
		"""
		self.step_count = 0
		self.curr_state = {}
		self.last_action = None
		self.curr_player_id = None
		self.light_cards = {role: [] for role in const.ALL_ROLE}
		self.judge = Judge(self.np_random)
		self.dealer = Dealer(self.np_random)
		self.players = [Player(num, self.np_random) for num in range(self.num_players)]
		self.round = Round(self.np_random, self.dealer, self.players)

		# todo: 发牌
		for player in self.players:
			self.set_role_by_position(player)
			self.dealer.deal_cards(player, 13)
			player.calc_score_cards_by_hand()

		# todo: 开始亮牌
		self.curr_player_id = self.select_player_id()
		self.round.curr_player_id = self.curr_player_id
		self.start_light_cards()

		# 构建首出玩家状态数据，默认首出一直为0号玩家
		state = self.get_state(self.curr_player_id)
		self.curr_state = state

		return self.curr_state, self.curr_player_id

	def start_light_cards(self):
		"""
		开始亮牌
		"""
		def get_yet_lp():
			"""
			获取已经亮的牌
			"""
			lp_cards = []
			for _, lp in self.light_cards.items():
				lp_cards.extend(lp)
			return lp_cards
		for curr_p in self.players:
			hand_cards = curr_p.curr_hand_cards
			cards = lp_policy(hand_cards, get_yet_lp())
			# 存在亮牌，则添加玩家亮牌
			if cards:
				curr_p.set_light_cards(cards)
				self.light_cards[curr_p.role] = cards

	def select_player_id(self):
		"""
		搜索下一位玩家
		"""
		next_p = {
			0: 1,
			1: 2,
			2: 3,
			3: 0
		}
		return 0 if self.last_action is None else next_p.get(self.curr_player_id)

	@staticmethod
	def set_role_by_position(curr_p):
		"""
		选择更新当前玩家位置
		"""
		curr_p.role = {
			0: "down",
			1: "right",
			2: "up",
			3: "left"
		}[curr_p.player_id]

	def step(self, action):
		"""
		迭代动作
		"""
		self.step_count += 1
		curr_p = self.players[self.curr_player_id]
		self.last_action = action
		self.remove_step_action(curr_p, action)

		# 记录历史出牌
		self.round.turn_cards.append(action)
		self.round.round_cards[curr_p.role].append(action)
		self.dealer.played_cards[curr_p.role].append(action)

		# 打完一轮则计算收分
		self.round.proceed_round(action)

		# 判断有些是否结束
		if not self.judge.judge_name(self.players) or not self.round.winner_id:
			# 未收牌，则按照当前操作顺序
			if not self.round.has_collect:
				self.curr_player_id = self.select_player_id()
				self.round.curr_player_id = self.curr_player_id
			# 收牌后，则收牌玩家变为首出
			else:
				self.curr_player_id = self.round.curr_player_id

			state = self.get_state(self.curr_player_id)
			self.curr_state = state
			return self.curr_state, self.curr_player_id

		# todo: 游戏结束
		state = self.get_state(self.curr_player_id)
		self.curr_state = state
		return self.curr_state, self.curr_player_id

	@staticmethod
	def remove_step_action(curr_p, action):
		"""
		判断是否为分牌，是则删除，反之
		"""
		# 删除当前出牌
		if action in curr_p.curr_hand_cards:
			curr_p.curr_hand_cards.remove(action)
		# 判断是否为分牌
		if action in curr_p.curr_score_cards:
			curr_p.curr_score_cards.remove(action)
		# 判断是否为亮牌
		if action in curr_p.light_cards:
			curr_p.light_cards.remove(action)
		# 添加玩家所打卡牌
		curr_p.played_cards.append(action)

	def get_legal_actions(self):
		"""
		获取玩家合法动作
		"""
		return self.curr_state['legal_actions']

	def get_state(self, curr_id):
		"""
		获取模型状态数据
		"""
		curr_p = self.players[curr_id]
		other_hand_cards = self.get_other_hand_cards(curr_p)
		state = self.round.get_state(curr_p, self.last_action, other_hand_cards, )
		return state

	def get_other_hand_cards(self, curr_p):
		"""
		获取其他玩家卡牌
		"""
		others_hand_cards = {'down': [], 'right': [], 'up': [], 'left': []}
		for other_p in self.players:
			if other_p.player_id == curr_p.player_id:
				continue
			others_hand_cards[other_p.role].extend(other_p.curr_hand_cards)
		return others_hand_cards

	@staticmethod
	def get_num_actions():
		"""
		卡牌动作数量
		"""
		return 54

	def get_player_id(self):
		"""
		获取当前玩家ID
		"""
		return self.round.curr_player_id

	def get_num_players(self):
		"""
		玩家数量
		"""
		return self.num_players

	def is_over(self):
		"""
		判断游戏是否结束
		"""
		return True if self.round.winner_id is not None else False