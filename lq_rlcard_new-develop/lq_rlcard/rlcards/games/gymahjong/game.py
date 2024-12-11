# -*- coding: utf-8 -*-

import numpy as np

from rlcards.const.mahjong import const
from rlcards.utils.mahjong.xxc import RuleXXC
from rlcards.games.mahjong.round import MahjongRound as Round
from rlcards.games.mahjong.judge import MahjongJudge as Judge
from rlcards.games.mahjong.player import MahjongPlayer as Player
from rlcards.games.mahjong.dealer import MahjongDealer as Dealer
from rlcards.const.mahjong.const import ActionType, OverType, CardType


class MahjongGame:
	"""
	麻将游戏流程包装类
	"""
	def __init__(self, allow_step_back=False):
		"""
		初始化麻将游戏参数
		"""
		self.judge = None
		self.round = None
		self.dealer = None
		self.players = None
		self.last_landlord_id = None
		self.num_players = const.PLAYERS_NUM
		self.allow_step_back = allow_step_back
		self.np_random = np.random.RandomState()

	def init_game(self):
		"""
		初始化游戏
		"""
		self.is_hz = True                   # 是否为荒庄
		self.curr_state = []  				# 玩家当前状态
		self.winner_id = None  				# 赢家ID
		self.curr_action = None  			# 当前操作动作
		self.curr_player = None  			# 当前玩家对象
		self.dynamics_reward = 0.1  		# 动态基础奖励(计算胡开奖励)

		self.judge = Judge(self.np_random)
		self.dealer = Dealer(self.np_random)
		self.players = [Player(i, self.np_random) for i in range(self.num_players)]
		self.round = Round(self.judge, self.dealer, self.num_players, self.np_random)

		# 游戏开局时，通过执行骰子，骰子点数最大者为庄
		# 游戏结束(胡开)时，下一局庄家ID为本局赢家ID
		# 游戏结束(流局)时，下一局庄家ID为本局庄家ID
		if self.last_landlord_id is None:
			# 开局定庄ID
			self.landlord_id = self.dealer.set_landlord_id(self.players)
		else:
			# 下一轮庄家ID
			self.landlord_id = self.last_landlord_id
		self.round.curr_player_id = self.landlord_id

		# 玩家发牌(13张)
		for player in self.players:
			self.dealer.deal_cards(player, 12)

		# 补牌(庄家)
		self.dealer.mo_cards(self.players[self.round.curr_player_id])

		# TODO: 是否能天胡和暗杠(庄家)
		valid_actions = self.judge.calc_operates_after_mo_pai(
			self.players[self.landlord_id],
			self.players,
			self.dealer,
			self.round.last_player_id
		)

		# 能天胡或暗杠时，有效动作为天胡或暗杠(庄家)
		self.round.valid_actions = valid_actions

		# 开局状态数据(庄家)
		state = self.get_state(self.round.curr_player_id)
		self.curr_state = state

		return self.curr_state, self.round.curr_player_id

	def step(self, action):
		"""
		TODO: 更新玩家状态数据
		"""
		# 当前操作玩家和执行动作
		self.curr_action = action
		self.curr_player = self.players[self.round.curr_player_id]

		# 判断冲锋鸡和冲锋乌骨鸡
		if isinstance(self.curr_action, int):
			if self.curr_action in [CardType.YAO_JI, CardType.WU_GU_JI]:
				self.round.deal_first_ji(self.curr_action, self.curr_player)

		# TODO: 流局(该局无赢家)
		# 桌牌为0，卡牌数量统计为0，则游戏结束
		if len(self.dealer.deck) == 0 and self.dealer.left_count == 0:
			self.curr_action = OverType.LIU_JU  # 流局
			self.winner_id = -1  # 无赢家
			self.last_landlord_id = self.landlord_id  # 流局时，庄家为上一位庄家

		# TODO: 计算玩家胡(判断玩家)
		# 胡开分为: 捡牌胡开和摸牌胡牌
		elif action == ActionType.KAI_HU:
			# 判断摸牌胡开
			if self.curr_player == self.dealer.curr_mo_player:
				hu_type, hu_path = RuleXXC.get_hu_type(
					self.curr_player,
					self.players,
					self.round.last_player_id,
					self.dealer.landlord_id,
					self.dealer.curr_mo_card,
					True
				)

				# 胡牌类型奖励
				if hu_type:
					self.players[self.round.curr_player_id].rewards += 0.4  # 胡开奖励
					self.winner_id = self.round.curr_player_id  # 当前操作玩家为赢家
					self.last_landlord_id = self.winner_id  # 本局赢家
					self.curr_action = OverType.HU_KAI  # 胡开
					self.is_hz = False  # 判断是否荒庄
					self.hu_type_rewards(hu_type)  # 摸牌胡开类型奖励

					# 胡牌信息
					hu_info = {
						"hu_card": self.dealer.curr_mo_card,
						"hu_type": hu_type,
						"action": action,
						"player_id": self.round.curr_player_id
					}

					# 添加胡牌信息
					self.players[self.round.curr_player_id].hu_infos.append(hu_info)

			# 判断捡牌胡开
			else:
				hu_type, hu_path = RuleXXC.get_hu_type(
					self.curr_player,
					self.players,
					self.round.last_player_id,
					self.dealer.landlord_id,
					self.dealer.table[-1]
				)

				# 胡牌类型奖励
				if hu_type:
					self.players[self.round.curr_player_id].rewards += 0.4  # 胡开奖励
					self.winner_id = self.round.curr_player_id  # 当前操作玩家为赢家
					self.last_landlord_id = self.winner_id  # 本局赢家
					self.curr_action = OverType.HU_KAI  # 胡开
					self.is_hz = False  # 判断是否荒庄
					self.hu_type_rewards(hu_type)

					# 胡牌信息
					hu_info = {
						"hu_card": self.dealer.table[-1],
						"hu_type": hu_type,
						"action": action,
						"player_id": self.round.curr_player_id
					}

					# 添加胡牌信息
					self.players[self.round.curr_player_id].hu_infos.append(hu_info)

		# TODO: 判断其他玩家操作
		else:
			# 未胡开或流局，则继续下一个操作
			self.round.proceed_round(self.players, action)

			# 状态数据
			state = self.get_state(self.round.curr_player_id)
			self.curr_state = state
			return self.curr_state, self.round.curr_player_id

		state = self.get_state(self.curr_player.player_id)
		self.curr_state = state
		return self.curr_state, self.curr_player.player_id

	def hu_type_rewards(self, hu_type):
		"""
		TODO: 胡牌类型奖励
		"""
		# 奖励倍率
		lv = 0.01
		# 计算胡牌分数和奖励
		for hu in hu_type:
			scores = const.HU_PAI_SCORES[hu]
			reward = (lv * scores)
			# 添加奖励
			self.dynamics_reward += reward

	def get_state(self, player_id):
		"""
		状态数据
		"""
		curr_player = self.players[player_id]
		other_hand_cards = self.get_others_curr_hand_cards(curr_player)
		hand_card_nums = [len(self.players[i].curr_hand_cards) for i in range(self.num_players)]
		state = self.round.get_state(self.players, player_id, other_hand_cards, hand_card_nums)

		return state

	def get_others_curr_hand_cards(self, player):
		"""
		计算其他玩家当前手牌
		"""
		others_index = []
		others_hand = []
		# 玩家ID为0
		if player.player_id == 0:
			others_hand.extend(self.players[(player.player_id + 1) % len(self.players)].curr_hand_cards)
			others_hand.extend(self.players[(player.player_id + 2) % len(self.players)].curr_hand_cards)
		# 玩家ID不为0时
		else:
			for p_id, p in enumerate(self.players):
				if p_id != player.player_id:
					others_index.append(p_id)
			others_index.sort(key=lambda idx: idx, reverse=False)
			others_hand.extend(self.players[others_index[0]].curr_hand_cards)
			others_hand.extend(self.players[others_index[1]].curr_hand_cards)

		return others_hand

	def get_legal_actions(self):
		"""
		TODO: 获取玩家合法动作
		"""
		# 无碰、杠等操作时，当前手牌为合法动作
		if self.curr_state['valid_actions'] == ['play']:
			self.curr_state['valid_actions'] = self.curr_state['actions']
			return self.curr_state['actions']

		# 有碰、杠等操作时，则碰、杠等操作为合法动作
		else:
			return self.curr_state['valid_actions']

	@staticmethod
	def get_num_actions():
		"""
		获取动作数量
		"""
		return 39

	def get_num_players(self):
		"""
		获取玩家数量
		"""
		return self.num_players

	def get_player_id(self):
		"""
		获取玩家ID
		"""
		return self.round.curr_player_id

	def is_over(self):
		"""
		TODO: 判断对局是否结束
		"""
		# 胡开(游戏结束，有赢家)
		if self.curr_action == OverType.HU_KAI and self.winner_id is not None:
			return True

		# 流局(游戏结束，无赢家)
		elif self.curr_action == OverType.LIU_JU and self.winner_id == -1:
			return True

		# 打牌(继续下一个操作流程)
		else:
			return False

	def round_over_xl_hz(self):
		"""
		血流红中一局结束，鸡分结算
		"""
		# 判断玩家操作
		for player in self.players:
			if not player:
				continue
			if player.player_id == self.winner_id:
				# 连庄次数
				player.lz += 1
			else:
				player.lz = 0

			# 判断玩家是否叫牌
			player.call_pai = RuleXXC.get_round_over_call_cards(player.piles, player.curr_hand_cards)

		# 结算玩家鸡分
		return self.round.do_check_over_xl_hz(self.players, self.dealer, self.is_hz)

	def round_over_fk(self, players, over_type):
		"""
		房开一局结束，鸡分结算
		"""
		# 判断是否流局(荒庄)
		# TODO: is_hz -> 是否荒庄
		self.is_hz = over_type == OverType.LIU_JU
		for player in players:
			if not player:
				continue
			if player.player_id == self.winner_id:
				# 连庄次数
				player.lz += 1
			else:
				player.lz = 0

			# 计算玩家叫牌类型
			player.call_pai = RuleXXC.get_round_over_call_cards(player.piles, player.curr_hand_cards)

		return self.round.do_check_over_fk(players, over_type)