# -*- coding: utf-8 -*-

import numpy as np

from rlcards.const.sytx import const
from rlcards.games.sytx.judge import SyJudge as Judge
from rlcards.games.sytx.round import SyRound as Round
from rlcards.games.sytx.player import SyPlayer as Player
from rlcards.games.sytx.dealer import SyDealer as Dealer


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
		self.landlord_id = None
		self.curr_player_id = None
		self.num_players = const.PLAYER_NUMS
		self.allow_step_back = allow_step_back
		self.np_random = np.random.RandomState()

		self.curr_state = {}  # 流程信息
		self.round_actions = []  # 一局游戏动作序列
		self.predict_infos = []  # 存储网络预测结果
		self.history_rounds = []  # 对局历史编码

	def init_game(self):
		"""
		初始化游戏
		"""
		self.curr_state = []  # 初始化状态
		self.round_actions = []  # 一局游戏动作序列
		self.judge = Judge(self.np_random)
		self.dealer = Dealer(self.np_random)
		self.players = [Player(num, self.np_random) for num in range(self.num_players)]
		self.round = Round(self.np_random, self.dealer, self.players)

		# 闲家初始操作为强攻、密、撒扑, 闲家开局选取操作[强攻、密、撒扑]
		self.landlord_id = self.dealer.set_regular_roles(self.players)
		for player in self.players:
			if player.role == "farmer":
				self.curr_player_id = player.player_id
				self.round.curr_player_id = self.curr_player_id

			self.dealer.deal_cards(player)

		# 开局判断水鱼或水鱼天下
		# 检测当前闲家手牌是否为水鱼或水鱼天下
		self.round.check_sy_tx(self.curr_player_id)
		# 开局水鱼天下，则重新初始化游戏
		if self.round.winner_id is not None:
			self.init_game()

		# 收集闲家预测状态数据
		state = self.get_state(self.curr_player_id)
		self.curr_state = state
		return self.curr_state, self.curr_player_id

	def step(self, action):
		"""
		迭代动作
		"""
		# 记录本轮所有操作动作
		self.round_actions.append((self.curr_player_id, action))
		# 记录所有对局操作动作
		self.predict_infos.append((self.curr_player_id, action))

		# todo: 动作流程判断
		# 闲家[强攻、密、撒扑]，庄家[走、杀、认、开]
		self.round.proceed_round(action)

		# 判断本轮游戏是否结束，当赢家ID不为None时，则表示本轮游戏结束
		if not self.judge.judge_name(self.round):
			# 否则，寻找下一位操作玩家ID，并根据提取的状态数据预测操作
			self.curr_player_id = self.round.select_next_id(self.curr_player_id)
			self.round.curr_player_id = self.curr_player_id
			# 更新状态数据
			state = self.get_state(self.curr_player_id)
			self.curr_state = state
			return self.curr_state, self.curr_player_id

		# 本轮游戏结束，将不再提取对应状态数据
		# 即使提取了状态数据，也不再传入网络模型
		# 收集赢家状态数据
		state = self.get_state(self.round.winner_id)
		self.curr_state = state
		# 存储当前对局所使用到的所有状态信息
		self.history_rounds.append(self.get_history_round_state())
		return self.curr_state, self.curr_player_id

	def get_history_round_state(self):
		"""
		对局已经结束的已经状态信息
		"""
		# 使用卡牌
		# 未使用卡牌
		# 所有合法动作
		# 历史合法动作
		# 选择合法动作
		history_round_state = {
			"winner_id": self.round.winner_id,
			"winner_role": self.players[self.round.winner_id].role,
			"winner_cards": self.calc_winner_cards(),
			"loser_cards": self.calc_loser_cards(),
			"used_by_cards": self.calc_used_by_cards(),
			"remain_cards": self.calc_remain_cards(),
			"lai_zi_cards": self.calc_lai_zi_cards(),
			"history_last_actions": self.calc_history_last_actions()
		}

		return history_round_state

	def calc_winner_cards(self):
		"""
		计算赢家卡牌
		"""
		return self.players[self.round.winner_id].curr_hand_cards

	def calc_loser_cards(self):
		"""
		计算输家卡牌
		"""
		loser_cards = []
		for curr_p in self.players:
			if curr_p.player_id == self.round.winner_id:
				continue
			loser_cards.extend(curr_p.curr_hand_cards)
		return loser_cards

	def calc_used_by_cards(self):
		"""
		计算已经使用的卡牌
		"""
		used_by_cards = []
		for curr_p in self.players:
			used_by_cards.extend(curr_p.curr_hand_cards)
		return used_by_cards

	def calc_lai_zi_cards(self):
		"""
		计算使用的癞子牌
		"""
		lai_zi_cards = []
		used_by_cards = self.calc_used_by_cards()
		for card in used_by_cards:
			if card in const.LAI_ZI:
				lai_zi_cards.append(card)
		return lai_zi_cards

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

	def calc_history_last_actions(self):
		"""
		计算上一个合法动作
		"""
		history_last_actions = []
		for idx, actions in enumerate(self.round_actions):
			history_last_actions.append(actions[1])
		return history_last_actions

	def get_state(self, curr_player_id):
		"""
		更新状态
		"""
		curr_player = self.players[curr_player_id]
		other_hand_cards = self.get_others_hand_cards(curr_player_id)
		hand_card_nums = [len(self.players[i].curr_hand_cards) for i in range(self.num_players)]

		# 打包状态数据
		state = self.round.get_state(
			curr_player,
			other_hand_cards,
			hand_card_nums,
			self.predict_infos,
			self.round_actions,
		)

		return state

	def get_legal_actions(self):
		"""
		合法动作
		"""
		return self.curr_state['actions']

	def get_others_hand_cards(self, curr_player_id):
		"""
		计算另一位玩家手牌
		"""
		next_id = self.round.select_next_id(curr_player_id)
		next_player = self.players[next_id]
		return next_player.curr_hand_cards

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
		# 判断游戏是否结束，赢家是否已经产生
		if self.round.winner_id is not None:
			return True
		# 否则，继续下一位玩家对象操作
		return False

	def calc_winner_counts(self):
		"""
		统计赢家次数
		"""
		if self.round.winner_id == 1:
			return "farmer"
		if self.round.winner_id == -1:
			return "draw"
		return "landlord"