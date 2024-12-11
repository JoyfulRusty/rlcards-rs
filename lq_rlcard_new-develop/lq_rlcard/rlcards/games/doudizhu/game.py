# -*- coding: utf-8 -*-

import functools
import numpy as np

from heapq import merge
from rlcards.const.doudizhu.const import CARD_RANK_STR, PLAYER_NUMS

from rlcards.games.doudizhu.round import DdzRound as Round
from rlcards.games.doudizhu.judge import DdzJudge as Judge
from rlcards.games.doudizhu.player import DdzPlayer as Player
from rlcards.games.doudizhu.utils import cards2str, ddz_sort_card_obj

class DdzGame:
	"""
	为envs提供怪物API来运行doudizhu并获取相应的状态信息
	"""
	def __init__(self, allow_step_back=False):
		"""
		初始化game属性参数
		"""
		self.allow_step_back = allow_step_back
		self.np_random = np.random.RandomState()
		self.num_players = PLAYER_NUMS

	def init_game(self):
		"""
		初始化游戏玩家和状态(state)
		:return: 第一个state和玩家id
		"""
		# 初始化变量
		self.winner_id = None
		self.hoistory = []

		# 初始化玩家
		self.players = [Player(num, self.np_random) for num in range(self.num_players)]

		# 初始化round和该局的landlord
		self.played_cards = [np.zeros((len(CARD_RANK_STR),), dtype=np.int32) for _ in range(self.num_players)]

		self.round = Round(self.np_random, self.played_cards)
		self.round.initiate(self.players)

		# 初始化judge
		self.judge = Judge(self.players, self.np_random)

		# 第一个state和第一个玩家
		player_id = self.round.current_player
		self.state = self.get_state(player_id)

		return self.state, player_id

	def step(self, action):
		"""
		进行step
		:param action(str): 斗地主的具体动作，例如： ‘3334’
		:return: 下一个玩家state和id
		"""
		if self.allow_step_back:
			pass

		player = self.players[self.round.current_player]

		# 判断玩家的操作(出牌和捡牌)，计算出上一位玩家
		self.round.proceed_round(player, action)

		# 也就是说上一位玩家的action不等于pass时
		# 说明它出牌了，现在需要计算
		if action != 'pass':
			self.judge.calc_playable_cards(player)

		if self.judge.judge_game(self.players, self.round.current_player):
			self.winner_id = self.round.current_player

		# 获取下一个玩家id
		next_id = (player.player_id + 1) % len(self.players)
		self.round.current_player = next_id

		# 获取下一个state
		state = self.get_state(next_id)
		self.state = state
		return state, next_id

	def step_back(self):
		"""
		回到之前的状态
		:return: bool -> True/False, 如果后退，则为真
		"""
		if not self.round.trace:
			return False

		# 无论任何情况下的step_back，winner_id都将始终为None
		self.winner_id = None

		# reverse round (反向回合)
		player_id, cards = self.round.step_back(self.players)

		# reverse player
		if cards != 'pass':
			self.players[player_id].played_cards = self.round.find_last_played_cards_in_trace(player_id)
		self.players[player_id].play_back()

		# reverse judger.played_cards if needed
		if cards != 'pass':
			self.judge.restore_playable_cards(player_id)

		self.state = self.get_state(self.round.current_player)
		return True

	def get_state(self, player_id):
		"""
		返回返家state
		:param player_id: 玩家id
		:return: (dict)玩家state
		"""
		player = self.players[player_id]
		other_hands = self._get_others_current_hand(player)
		num_cards_left = [len(self.players[i].current_hand) for i in range(self.num_players)]

		if self.is_over():
			actions = []
		else:
			actions = list(player.available_actions(self.round.greater_player, self.judge))

		state = player.get_state(self.round.public, other_hands, num_cards_left, actions)

		return state

	@staticmethod
	def get_num_actions():
		"""
		返回玩家全部的动作抽象
		:return: 斗地主抽象动作总数
		"""
		return 27472

	def get_player_id(self):
		"""
		返回当前玩家id
		:return: 当前玩家id
		"""
		return self.round.current_player

	def get_num_players(self):
		"""
		返回斗地主的玩家数量
		:return: 斗地主玩家数量
		"""
		return self.num_players

	def is_over(self):
		"""
		是否结束
		:return: bool -> True(over) / False(not over)
		"""
		if self.winner_id is None:
			return False
		return True

	def _get_others_current_hand(self, player):
		"""
		获取其他玩家的手牌
		:param player: 当前玩家
		:return:
		"""
		player_up = self.players[(player.player_id + 1) % len(self.players)]
		player_down = self.players[(player.player_id - 1) % len(self.players)]
		others_hand = merge(player_up.current_hand, player_down.current_hand,
							key=functools.cmp_to_key(ddz_sort_card_obj))
		return cards2str(others_hand)