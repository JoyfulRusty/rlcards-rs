# -*- coding: utf-8 -*-

import numpy as np

from rlcards.envs.env import Env
from collections import Counter, OrderedDict
from rlcards.games.sytx.game import SyGame as Game
from rlcards.const.sytx.const import LAI_ZI, FarmerAction
from rlcards.games.sytx.utils import new_card_encoding_dict, new_action_encoding_dict, new_lz_encoding_dict


class SyEnv(Env):
	"""
	打妖怪游戏环境
	"""

	def __init__(self, config):
		"""
		初始化打妖怪属性参数
		"""
		self.name = 'sytx'
		self.game = Game()
		super().__init__(config, self.game)

		self.action_id = new_action_encoding_dict
		self.decode_id = new_action_encoding_dict
		self.state_shape = [[664], [664]]  # conv [664], [664] -> [4415, 4415]
		self.action_shape = [[9] for _ in range(self.num_players)]
		self.decode_action_id = {self.decode_id[key]: key for key in self.decode_id.keys()}

	def extract_state(self, state):
		"""
		抽取状态编码
		"""
		# 状态编码state
		obs = self.get_obs(state)
		extracted_state = {
			'obs': obs,
			'z_obs': obs.reshape(9, 381).astype(np.float32),
			'actions': self._get_legal_actions(),
			'row_state': state,
			'row_legal_actions': self.get_row_legal_actions(state['actions']),
			'action_record': self.action_recorder
		}

		# print("输出动作记录: ", self.action_recorder)

		return extracted_state

	def get_obs(self, state):
		"""
		根据当前角色选择状态编码
		"""
		if state["role"] == "farmer":
			return self.get_farmer_obs(state)
		return self.get_landlord_obs(state)

	def get_landlord_obs(self, state):
		"""
		庄家单局回合状态编码
		"""
		curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
		curr_big_cards = self.encode_cards(state['curr_hand_cards'][:2])
		curr_small_cards = self.encode_cards(state['curr_hand_cards'][2:])
		other_hand_cards = self.encode_hidden_cards(state['other_hand_cards'], state)
		other_big_cards = self.encode_hidden_cards(state['other_hand_cards'][:2], state)
		other_small_cards = self.encode_hidden_cards(state['other_hand_cards'][2:], state)
		remain_cards = self.encode_cards(state['remain_cards'])
		lai_zi_cards = self.encode_lai_zi_cards(state["used_by_cards"])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		action_roles = self.encode_landlord_or_farmer(state['role'])

		# 对一整局游戏所使用的状态信息编码，十局后则清空，重新记录
		history_rounds = self.encode_history_round()

		obs = np.hstack((
			curr_hand_cards,
			curr_big_cards,
			curr_small_cards,
			other_hand_cards,
			other_big_cards,
			other_small_cards,
			remain_cards,
			lai_zi_cards,
			last_action,
			legal_actions,
			round_actions,
			action_roles,
			history_rounds
		))

		return obs

	def get_farmer_obs(self, state):
		"""
		闲家单局回合状态编码
		"""
		curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
		curr_big_cards = self.encode_cards(state['curr_hand_cards'][:2])
		curr_small_cards = self.encode_cards(state['curr_hand_cards'][2:])
		other_hand_cards = self.encode_cards(state['other_hand_cards'])
		other_big_cards = self.encode_cards(state['other_hand_cards'][:2])
		other_small_cards = self.encode_cards(state['other_hand_cards'][2:])
		remain_cards = self.encode_cards(state['remain_cards'])
		lai_zi_cards = self.encode_lai_zi_cards(state["used_by_cards"])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		action_roles = self.encode_landlord_or_farmer(state['role'])

		# 对一整局游戏所使用的状态信息编码，十局后则清空，重新记录
		history_rounds = self.encode_history_round()

		obs = np.hstack((
			curr_hand_cards,
			curr_big_cards,
			curr_small_cards,
			other_hand_cards,
			other_big_cards,
			other_small_cards,
			remain_cards,
			lai_zi_cards,
			last_action,
			legal_actions,
			round_actions,
			action_roles,
			history_rounds
		))

		return obs

	@staticmethod
	def get_row_legal_actions(actions):
		"""
		处理合法动作
		"""
		# 当前动作为非法动作时，返回[]
		if not actions:
			return []
		# 当前动作为合法动作，则遍历当前的合法动作
		return [action for action in actions]

	def get_payoffs(self):
		"""
		奖励
		"""
		return self.game.judge.judge_payoffs(self.game)

	def _decode_action(self, action_id):
		"""
		解码动作
		"""
		if isinstance(action_id, dict):
			return None

		action = self.decode_action_id[action_id]
		if action_id < 10:
			legal_cards = self.game.get_legal_actions()
			for card in legal_cards:
				if card == action:
					action = card
					break

		return action

	def _get_legal_actions(self):
		"""
		合法动作
		"""
		legal_action_id = {}
		legal_actions = self.game.get_legal_actions()
		if legal_actions:
			for action in legal_actions:
				legal_action_id[self.action_id[action]] = None

		return OrderedDict(legal_action_id)

	@staticmethod
	def get_action_feature(action_id):
		"""
		获取动作特征
		"""
		if not action_id or isinstance(action_id, dict):
			return np.zeros(9, dtype=np.float32)
		matrix = np.zeros(9, dtype=np.float32)
		matrix[action_id] = 1
		return matrix.flatten('A')

	@staticmethod
	def encode_cards(cards):
		"""
		卡牌编码
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards:
			return np.zeros(55, dtype=np.float32)
		matrix = np.zeros((4, 13), dtype=np.float32)
		ghost_matrix = np.zeros(3, dtype=np.float32)
		cards_dict = Counter(cards)
		for card, num_times in cards_dict.items():
			if card < 414:
				# [花色, 牌值]
				matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1
			# [鬼点/癞子]
			elif card == 515:
				ghost_matrix[0] = 1
			elif card == 516:
				ghost_matrix[1] = 1
			elif card == 517:
				ghost_matrix[2] = 1

		matrix = matrix.flatten('A')

		return np.concatenate((matrix, ghost_matrix))

	@staticmethod
	def encode_hidden_cards(cards, state):
		"""
		闲家密牌时，则对庄家隐藏当前卡牌信息
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards or state["last_action"] == FarmerAction.MI:
			return np.zeros(55, dtype=np.float32)
		matrix = np.zeros((4, 13), dtype=np.float32)
		ghost_matrix = np.zeros(3, dtype=np.float32)
		cards_dict = Counter(cards)
		for card, num_times in cards_dict.items():
			if card < 414:
				# [花色, 牌值]
				matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1
			# [鬼点/癞子]
			elif card == 515:
				ghost_matrix[0] = 1
			elif card == 516:
				ghost_matrix[1] = 1
			elif card == 517:
				ghost_matrix[2] = 1

		matrix = matrix.flatten('A')

		return np.concatenate((matrix, ghost_matrix))

	@staticmethod
	def calc_ghost_cards(tmp_cards):
		"""
		计算大小王与癞子牌
		"""
		ghost_cards = []
		for card in tmp_cards:
			if card in LAI_ZI:
				ghost_cards.append(card)
				tmp_cards.remove(card)
		return tmp_cards, ghost_cards

	@staticmethod
	def encode_lai_zi_cards(used_by_cards):
		"""
		编码癞子牌
		"""
		# 统计当前使用的卡牌中是否存在癞子牌
		lai_zi_cards = []
		for card in used_by_cards:
			if card in LAI_ZI:
				lai_zi_cards.append(card)

		# 不存在癞子，则全部返回0.0
		if not lai_zi_cards:
			return np.zeros(3, dtype=np.float32)

		# 存在癞子时，则编码对应的卡牌
		matrix = np.zeros(3, dtype=np.float32)
		for lz in lai_zi_cards:
			matrix[new_lz_encoding_dict[lz]] = 1

		matrix = matrix.flatten('A')

		return matrix

	def encode_last_action(self, last_action):
		"""
		编码上一个动作
		"""
		if not last_action:
			return np.zeros(9, dtype=np.float32)
		matrix = np.zeros(9, dtype=np.float32)
		matrix[self.action_id[last_action]] = 1

		return matrix.flatten('A')

	def encode_legal_actions(self, actions):
		"""
		编码合法动作
		"""
		if not actions:
			return np.zeros(27, dtype=np.float32)

		matrix = np.zeros((3, 9), dtype=np.float32)
		for idx, action in enumerate(actions):
			matrix[idx][self.action_id[action]] = 1

		return matrix.flatten('A')

	def encode_round_actions(self, round_actions):
		"""
		编码本轮动作序列
		"""
		if not round_actions:
			return np.zeros(27, dtype=np.float32)
		matrix = np.zeros((3, 9), dtype=np.float32)
		for idx, action in enumerate(round_actions):
			matrix[idx][self.action_id[action[1]]] = 1

		return matrix.flatten('A')

	@staticmethod
	def encode_landlord_or_farmer(landlord_or_farmer, is_draw=False):
		"""
		判断当前玩家是否为庄家
		"""
		if not landlord_or_farmer or is_draw:
			return np.zeros(18, dtype=np.float32)
		matrix = np.zeros((2, 9), dtype=np.float32)
		if landlord_or_farmer == 'landlord':
			matrix[0][:] = 1
		else:
			matrix[1][:] = 1
		return matrix.flatten('A')

	def encode_history_round(self):
		"""
		编码历史对局
		"""
		# 对局信息记录为空时，则全部返回零
		if not self.game.history_rounds:
			return np.zeros(10 * 296, dtype=np.float32)
		# 十局历史记录编码
		collect_remain_by_matrix = np.zeros((10, 296), dtype=np.float32)
		for idx, history_round in enumerate(self.game.history_rounds):
			winner_id_round = self.encode_landlord_or_farmer(history_round["winner_role"], history_round["winner_id"] == -1)
			winner_big_cards = self.encode_cards(history_round["winner_cards"][:2])
			winner_small_cards = self.encode_cards(history_round["winner_cards"][2:])
			loser_big_cards = self.encode_cards(history_round["loser_cards"][:2])
			loser_small_cards = self.encode_cards(history_round["loser_cards"][2:])
			remain_by_matrix = self.encode_cards(history_round["remain_cards"])
			lai_zi_by_matrix = self.encode_lai_zi_cards(history_round["lai_zi_cards"])

			package_matrix = np.hstack((
				winner_id_round,
				winner_big_cards,
				winner_small_cards,
				loser_big_cards,
				loser_small_cards,
				remain_by_matrix,
				lai_zi_by_matrix,
			))

			collect_remain_by_matrix[idx] = package_matrix.flatten('A')

		if len(self.game.history_rounds) >= 10:
			self.game.history_rounds.clear()

		return collect_remain_by_matrix.flatten('A')