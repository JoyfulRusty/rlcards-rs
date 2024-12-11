# -*- coding: utf-8 -*-

import numpy as np

from rlcards.const.sytx_gz import const
from rlcards.games.sytx_gz.rules import Rules
from rlcards.const.sytx_gz.const import FarmerAction
from rlcards.games.sytx_gz.judge import SyJudge as Judge
from rlcards.games.sytx_gz.round import SyRound as Round
from rlcards.games.sytx_gz.player import SyPlayer as Player
from rlcards.games.sytx_gz.dealer import SyDealer as Dealer
from rlcards.games.sytx_gz.allocat import get_better_combine_by_3


class SyGame:
	"""
	水鱼游戏流程
	"""
	def __init__(self, allow_step_back=False):
		"""
		初始化游戏参数
		"""
		self.judge = None
		self.round = None
		self.dealer = None
		self.players = None
		self.curr_player_id = None
		self.num_players = const.PLAYER_NUMS
		self.allow_step_back = allow_step_back
		self.np_random = np.random.RandomState()

		self.curr_state = {}  # 流程信息
		self.round_actions = []  # 一局游戏动作序列
		self.history_actions = []  # 存储网络预测结果
		self.history_rounds = []  # 对局历史编码

	def init_game(self):
		"""
		初始化游戏
		"""
		self.curr_state = []
		self.round_actions = []
		self.judge = Judge(self.np_random)
		self.dealer = Dealer(self.np_random)
		self.players = [Player(num, self.np_random) for num in range(self.num_players)]
		self.round = Round(self.np_random, self.dealer, self.players)

		# 设置庄闲训练模型角色
		# 闲家初始操作为强攻、密、撒扑, 闲家开局选取操作[强攻、密、撒扑]
		self.dealer.set_regular_roles(self.players)

		# todo: 首出为闲家
		for player in self.players:
			# 闲家首次选择操作，操作完后，庄闲分扑明牌(3x)
			if player.role == "farmer":
				self.curr_player_id = player.player_id
				self.round.curr_player_id = self.curr_player_id
				# 添加闲家亮牌分扑
				player.receive_combine_by_3(get_better_combine_by_3(player.bright_cards))

			# 发牌[3x明牌 + 1x暗牌]
			self.dealer.deal_cards(player)

		# 构建收集闲家首次预测状态数据
		state = self.get_state(self.curr_player_id)
		self.curr_state = state

		return self.curr_state, self.curr_player_id

	def step(self, action):
		"""
		迭代动作
		"""
		# todo: 庄闲分扑
		# 闲家选择完第一次动作后，开始分扑明牌
		if not self.round_actions:
			self.get_all_combine_3()

		# 存储历史动作
		self.memory_history_actions(action)

		# todo: 动作流程判断
		# 闲家[强攻、密、撒扑、信、反]，庄家[走、杀、认、开]
		self.round.proceed_round(action)

		# 判断本轮游戏是否结束，当赢家ID不为None时，则表示本轮游戏结束
		if not self.judge.judge_name(self.round):
			# 否则，寻找下一位操作玩家ID，并根据收集对应状态数据
			self.curr_player_id = self.round.select_next_id(self.curr_player_id)
			self.round.curr_player_id = self.curr_player_id

			# 更新状态数据
			state = self.get_state(self.curr_player_id)
			self.curr_state = state

			return self.curr_state, self.curr_player_id

		# 收集赢家状态数据
		state = self.get_state(self.round.winner_id)
		self.curr_state = state

		# 存储当前对局所使用到的所有状态信息
		self.history_rounds.append(self.get_history_round_state())
		self.history_actions.append(self.get_history_round_actions())

		return self.curr_state, self.curr_player_id

	def memory_history_actions(self, action):
		"""
		存储历史动作
		"""
		# 记录本次对局使用的所有动作
		self.round_actions.append((self.curr_player_id, action))

	def get_history_round_actions(self):
		"""
		历史对局动作
		"""
		if not self.round_actions:
			return []
		round_actions = []
		for actions in self.round_actions:
			round_actions.append(actions[1])
		return round_actions

	def get_all_combine_3(self):
		"""
		庄闲开始分扑明牌
		"""
		for curr_p in self.players:
			if curr_p.role == "farmer":
				continue
			curr_p.receive_combine_by_3(get_better_combine_by_3(curr_p.bright_cards))

	def get_history_round_state(self):
		"""
		对局已经结束的已经状态信息
		"""
		history_round_state = {
			"winner_id": self.round.winner_id,
			"winner_role": self.calc_winner_role(),
			"landlord_cards": self.calc_landlord_cards(),
			"farmer_cards": self.calc_farmer_cards(),
			"compare_big_cards": self.compare__big_cards(),
			"compare_small_cards": self.compare_small_cards(),
			"remain_cards": self.calc_remain_cards(),
		}

		return history_round_state

	def calc_winner_role(self):
		"""
		计算赢家角色
		"""
		if self.round.winner_id == -1:
			return "draw"
		for curr_p in self.players:
			if curr_p.player_id == self.round.winner_id:
				return curr_p.role

	def calc_landlord_cards(self):
		"""
		计算赢家卡牌
		"""

		return self.players[self.round.winner_id].curr_hand_cards

	def calc_farmer_cards(self):
		"""
		计算输家卡牌
		"""
		for curr_p in self.players:
			if curr_p.player_id == self.round.winner_id:
				continue
			return curr_p.curr_hand_cards

		return []

	def compare__big_cards(self):
		"""
		比较赢家大牌
		"""
		landlord_big_cards = self.players[0].curr_hand_cards[:2]
		farmer_big_cards = self.players[1].curr_hand_cards[:2]
		compare_res = Rules.compare_one_combine(farmer_big_cards, landlord_big_cards)
		# 庄家大扑大
		if compare_res == 4:
			return "landlord"
		# 闲家大扑大
		elif compare_res == 3:
			return "farmer"
		return "draw"

	def compare_small_cards(self):
		"""
		比较赢家小牌
		"""
		landlord_big_cards = self.players[0].curr_hand_cards[2:]
		farmer_big_cards = self.players[1].curr_hand_cards[2:]
		compare_res = Rules.compare_one_combine(farmer_big_cards, landlord_big_cards)
		# 庄家大扑大
		if compare_res == 4:
			return "landlord"
		# 闲家大扑大
		elif compare_res == 3:
			return "farmer"
		return "draw"

	def calc_remain_cards(self):
		"""
		计算未使用卡牌
		"""
		return self.round.get_remain_cards()

	@staticmethod
	def calc_legal_action(state):
		"""
		计算能够选择的合法动作
		"""
		return state["actions"]

	def get_state(self, curr_id):
		"""
		更新状态
		"""
		curr_player = self.players[curr_id]
		other_hand_cards = self.get_others_hand_cards(curr_id)
		# 打包状态数据
		state = self.round.get_state(
			curr_player,
			other_hand_cards,
			self.round_actions,
			self.history_actions
		)
		return state

	def get_legal_actions(self):
		"""
		合法动作
		"""
		return self.curr_state['actions']

	def get_others_hand_cards(self, curr_id):
		"""
		计算另一位玩家手牌
		"""
		# 闲家看不见庄家亮牌
		if not self.round_actions:
			return []

		# 闲家撒扑时，庄家能看见闲家亮牌
		elif self.round_actions[0][1] == FarmerAction.SP:
			# todo: 庄可见闲
			if self.players[curr_id].role == "landlord":
				farmer = self.players[self.round.select_next_id(curr_id)]
				# 撒扑时，庄家可见闲家的亮牌，返回其亮牌
				return farmer.bright_cards

			# todo: 闲不可见庄
			# 闲家返回[]，闲家看不见庄家亮牌
			return []

		# 其他动作则返回[]
		else:
			return []

	@staticmethod
	def get_num_actions():
		"""
		返回抽象动作数量
		"""
		return 9

	def get_player_id(self):
		"""
		当前操作ID
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

	def calc_winner_counts(self):
		"""
		统计赢家次数
		"""

		if self.round.winner_id == -1:
			return "draw"
		return "farmer" if self.round.winner_id == 1 else "landlord"