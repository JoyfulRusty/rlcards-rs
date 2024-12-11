# -*- coding: utf-8 -*-

import numpy as np

from collections import Counter, OrderedDict

from rlcards.const.sytx.const import FarmerAction, LAI_ZI
from rlcards.games.sytx.utils import new_card_encoding_dict, new_lz_encoding_dict, new_action_encoding_dict


class SyState:
	"""
	水鱼状态编码
	"""
	def __init__(self):
		"""
		初始化参数
		"""
		self.action_id = new_action_encoding_dict
		self.history_rounds = []
		self.predict_infos = []

	@staticmethod
	def build_state(state):
		"""
		构建单回合对局状态数据
		"""
		new_state = {
			"actions": state["actions"],
			"self": state["self"],
			"is_landlord": state["is_landlord"],
			"is_farmer": state["is_farmer"],
			"is_winner": state["is_winner"],
			"curr_hand_cards": state["curr_hand_cards"],
			"other_hand_cards": state["other_hand_cards"],
			"hand_cards_nums": state["hand_cards_nums"],
			"round_actions": state["round_actions"],
			"predict_infos": state["predict_infos"],
			"remain_cards": state["remain_cards"],
			"used_by_cards": state["used_by_cards"]
		}

		return new_state

	@staticmethod
	def build_round_state(round_state):
		"""
		构建整局的所有操作数据
		"""
		new_round_state = {
			"winner_id": round_state["winner_id"],
			"winner_cards": round_state["winner_cards"],
			"loser_cards": round_state["loser_cards"],
			"used_by_cards": round_state["used_by_cards"],
			"remain_cards": round_state["remain_cards"],
			"all_legal_actions": round_state["all_legal_actions"],
			"history_last_actions": round_state["history_last_actions"],
		}

		return new_round_state

	def extract_state(self, state):
		"""
		抽取状态编码
		"""
		# 状态编码state
		obs = self.get_obs(state)
		extracted_state = {
			'obs': obs,
			'z_obs': obs.reshape(15, 269).astype(np.float32),
			'actions': self.get_legal_actions(state),
			'row_state': state,
			'row_legal_actions': self.get_row_legal_actions(state['actions']),
		}

		return extracted_state

	def get_legal_actions(self, state):
		"""
		计算合法动作
		"""
		legal_action_id = {}
		legal_actions = state["actions"]
		if legal_actions:
			for action in legal_actions:
				legal_action_id[self.action_id[action]] = None

		return OrderedDict(legal_action_id)

	@staticmethod
	def get_row_legal_actions(actions):
		"""
		合法动作
		"""
		if not actions:
			return []
		return [action for action in actions]

	def decode_actions(self, state, action_id):
		"""
		解码动作
		"""
		if isinstance(action_id, dict):
			return
		decode_action = self.action_id[action_id]
		if action_id < 10:
			legal_actions = self.get_legal_actions(state)
			for action in legal_actions:
				if action == decode_action:
					decode_action = action
					break
		return decode_action

	def get_obs(self, state):
		"""
		单局回合状态数据编码
		"""
		curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
		curr_big_cards = self.encode_cards(state['curr_hand_cards'][:2])
		curr_small_cards = self.encode_cards(state['curr_hand_cards'][2:])
		other_hand_cards = self.encode_hidden_cards(state['other_hand_cards'], state)
		other_big_cards = self.encode_hidden_cards(state['other_hand_cards'][:2], state)
		other_small_cards = self.encode_hidden_cards(state['other_hand_cards'][2:], state)
		remain_cards = self.encode_cards(state['remain_cards'])
		used_by_cards = self.encode_cards(state["used_by_cards"])
		lai_zi_cards = self.encode_lai_zi_cards(state["used_by_cards"])
		hand_card_nums = self.encode_nums(state['hand_card_nums'])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		predict_actions = self.encode_predict_infos(state['predict_infos'])
		is_landlord = self.encode_landlord_and_farmer(state["is_landlord"])
		is_farmer = self.encode_landlord_and_farmer(state["is_farmer"])

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
			used_by_cards,
			lai_zi_cards,
			hand_card_nums,
			last_action,
			legal_actions,
			round_actions,
			predict_actions,
			is_landlord,
			is_farmer,
			history_rounds
		))

		return obs

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

		matrix = matrix.flatten('F')

		return np.concatenate((matrix, ghost_matrix))

	@staticmethod
	def encode_hidden_cards(cards, state):
		"""
		闲家密牌时，则对庄家隐藏当前卡牌信息
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards or (state["last_action"] == FarmerAction.MI and not state["is_landlord"]):
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

		matrix = matrix.flatten('F')

		return np.concatenate((matrix, ghost_matrix))

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

		matrix = matrix.flatten('F')

		return matrix

	@staticmethod
	def encode_nums(hand_card_nums):
		"""
		卡牌数量编码，破产将不再进行编码，全部置为0
		"""
		matrix = np.zeros((2, 4), dtype=np.float32)
		for idx, nums in enumerate(hand_card_nums):
			matrix[idx][nums - 1] = 1

		return matrix.flatten('F')

	def encode_last_action(self, last_action):
		"""
		编码上一个动作
		"""
		if not last_action:
			return np.zeros(9, dtype=np.float32)
		matrix = np.zeros(9, dtype=np.float32)
		matrix[self.action_id[last_action]] = 1

		return matrix.flatten('F')

	def encode_legal_actions(self, actions):
		"""
		编码合法动作
		"""
		if not actions:
			return np.zeros(9, dtype=np.float32)

		matrix = np.zeros(9, dtype=np.float32)
		for action in actions:
			matrix[self.action_id[action]] = 1

		return matrix.flatten('F')

	@staticmethod
	def encode_winner_id(winner_id):
		"""
		编码赢家ID
		"""
		if winner_id is None or winner_id == -1:
			return np.zeros(2, dtype=np.float32)
		winner_matrix = np.zeros(2, dtype=np.float32)
		if winner_id == 0:
			winner_matrix[0] = 1
			return winner_matrix
		winner_matrix[winner_id] = 1
		return winner_matrix

	def encode_round_actions(self, round_actions):
		"""
		编码本轮动作序列
		"""
		if not round_actions:
			return np.zeros(9, dtype=np.float32)
		matrix = np.zeros(9, dtype=np.float32)
		for idx, action in enumerate(round_actions):
			matrix[self.action_id[action[1]]] = 1

		return matrix.flatten('F')

	def encode_predict_infos(self, predict_infos):
		"""
		编码所有对局动作序列
		"""
		# 当预测信息为空时或总预测动作次数小于10时，则返回全0
		if not predict_infos or len(predict_infos) < 10:
			return np.zeros(9, dtype=np.float32)

		matrix = np.zeros(9, dtype=np.float32)
		# 每次对九个连续动作进行编码
		for idx, predict_action in enumerate(predict_infos[-9:]):
			if matrix[self.action_id[predict_action[1]]] == 1:
				continue
			matrix[self.action_id[predict_action[1]]] = 1

		# 编码一次，则清空存储数据
		self.predict_infos.clear()

		return matrix.flatten('F')

	@staticmethod
	def encode_landlord_and_farmer(landlord_or_farmer):
		"""
		判断当前玩家是否为庄家
		"""
		if not landlord_or_farmer:
			return np.zeros(9, dtype=np.float32)
		return np.ones(9, dtype=np.float32)

	def encode_history_round(self):
		"""
		编码历史对局
		"""
		# 对局信息记录为空时，则全部返回零
		if not self.history_rounds:
			return np.zeros(10 * 353, dtype=np.float32)
		# 十局历史记录编码
		collect_remain_by_matrix = np.zeros((10, 353), dtype=np.float32)
		for idx, history_round in enumerate(self.history_rounds):
			winner_id_round = self.encode_winner_id(history_round["winner_id"])
			winner_big_cards = self.encode_cards(history_round["winner_cards"][:2])
			winner_small_cards = self.encode_cards(history_round["winner_cards"][2:])
			loser_big_cards = self.encode_cards(history_round["loser_cards"][:2])
			loser_small_cards = self.encode_cards(history_round["loser_cards"][2:])
			used_by_matrix = self.encode_cards(history_round["used_by_cards"])
			remain_by_matrix = self.encode_cards(history_round["remain_cards"])
			lai_zi_by_matrix = self.encode_lai_zi_cards(history_round["lai_zi_cards"])
			all_legal_actions_matrix = self.encode_legal_actions(history_round["all_legal_actions"])
			history_last_actions_matrix = self.encode_legal_actions(history_round["history_last_actions"])

			package_matrix = np.hstack((
				winner_id_round,
				winner_big_cards,
				winner_small_cards,
				loser_big_cards,
				loser_small_cards,
				used_by_matrix,
				remain_by_matrix,
				lai_zi_by_matrix,
				all_legal_actions_matrix,
				history_last_actions_matrix
			))

			collect_remain_by_matrix[idx] = package_matrix.flatten('F')

		if len(self.history_rounds) >= 10:
			self.history_rounds.clear()

		return collect_remain_by_matrix.flatten('F')

sy_state = SyState()