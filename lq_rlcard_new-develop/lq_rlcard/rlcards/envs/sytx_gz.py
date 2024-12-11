# -*- coding: utf-8 -*-

import itertools
import numpy as np

from rlcards.envs.env import Env
from collections import Counter, OrderedDict
from rlcards.games.sytx_gz.game import SyGame as Game
from rlcards.games.sytx_gz.utils import new_card_encoding_dict, new_action_encoding_dict


class SyEnv(Env):
	"""
	打妖怪游戏环境
	"""

	def __init__(self, config):
		"""
		初始化打妖怪属性参数
		"""
		self.name = 'sytx_gz'
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
			'z_obs': obs.reshape(9, 427).astype(np.float32),
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
		curr_hand_cards = self.encode_cards(state['bright_cards'])
		curr_big_and_small_cards = self.encode_big_or_small_bright_cards(state['bright_cards'])
		curr_combine_cards = self.encode_cards(state['combine_cards'])
		compare_big_cards = self.compare_farmer_and_landlord_big_cards(state["compare_big_cards"])
		other_hand_cards = self.encode_cards(state['other_hand_cards'])
		other_big_and_small_cards = self.encode_big_or_small_bright_cards(state['other_hand_cards'])
		remain_cards = self.encode_cards(state['remain_cards'])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		history_actions = self.encode_history_actions(state['history_actions'])
		action_roles = self.encode_landlord_or_farmer(state['role'])

		# 对一整局游戏所使用的状态信息编码，十局后则清空，重新记录
		history_rounds = self.encode_history_round()

		obs = np.hstack((
			curr_hand_cards,
			curr_big_and_small_cards,
			curr_combine_cards,
			compare_big_cards,
			other_hand_cards,
			other_big_and_small_cards,
			remain_cards,
			last_action,
			legal_actions,
			round_actions,
			history_actions,
			action_roles,
			history_rounds
		))

		return obs

	def get_farmer_obs(self, state):
		"""
		闲家单局回合状态编码
		"""
		curr_hand_cards = self.encode_cards(state['bright_cards'])
		curr_big_and_small_cards = self.encode_big_or_small_bright_cards(state['bright_cards'])
		curr_combine_cards = self.encode_cards(state['combine_cards'])
		compare_big_cards = self.compare_farmer_and_landlord_big_cards(state["compare_big_cards"])
		other_hand_cards = self.encode_cards(state['other_hand_cards'])
		other_big_and_small_cards = self.encode_big_or_small_bright_cards(state['other_hand_cards'])
		remain_cards = self.encode_cards(state['remain_cards'])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		history_actions = self.encode_history_actions(state['history_actions'])
		action_roles = self.encode_landlord_or_farmer(state['role'])

		# 对一整局游戏所使用的状态信息编码，十局后则清空，重新记录
		history_rounds = self.encode_history_round()

		obs = np.hstack((
			curr_hand_cards,
			curr_big_and_small_cards,
			curr_combine_cards,
			compare_big_cards,
			other_hand_cards,
			other_big_and_small_cards,
			remain_cards,
			last_action,
			legal_actions,
			round_actions,
			history_actions,
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
	def combine_bright_cards(bright_cards):
		"""
		明牌x3组合
		"""
		combine_cards = itertools.permutations(bright_cards, 3)
		new_combine_cards = [list(cards) for cards in combine_cards]
		return new_combine_cards

	def encode_big_or_small_bright_cards(self, bright_cards):
		"""
		编码亮牌3X3组合
		"""
		if not bright_cards:
			return np.zeros(288, dtype=np.float32)

		# 三种组合，编码成9张
		all_combine_matrix = np.zeros((6, 48), dtype=np.float32)

		# 使用[4 x 12]矩阵来进行编码
		for idx, cards in enumerate(self.combine_bright_cards(bright_cards)):
			matrix = np.zeros((4, 12), dtype=np.float32)
			cards_dict = Counter(cards)
			for card, num_times in cards_dict.items():
				matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1

			all_combine_matrix[idx] = matrix.flatten('A')

		return all_combine_matrix.flatten('A')

	@staticmethod
	def encode_cards(cards):
		"""
		卡牌编码
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards:
			return np.zeros(48, dtype=np.float32)
		matrix = np.zeros((4, 12), dtype=np.float32)
		cards_dict = Counter(cards)
		for card, num_times in cards_dict.items():
			matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1

		matrix = matrix.flatten('A')

		return matrix

	@staticmethod
	def encode_big_and_small_cards(cards):
		"""
		编码大小扑卡牌
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards:
			return np.zeros(48, dtype=np.float32)
		matrix = np.zeros((4, 12), dtype=np.float32)
		cards_dict = Counter(cards)
		for card, num_times in cards_dict.items():
			matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1

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
			return np.zeros(36, dtype=np.float32)
		matrix = np.zeros((4, 9), dtype=np.float32)
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

	@staticmethod
	def compare_farmer_and_landlord_big_cards(compare_big_cards):
		"""
		todo: 比较亮牌后，庄闲家确定大扑大小比较
		"""
		if not compare_big_cards or compare_big_cards == "draw":
			return np.zeros(18, dtype=np.float32)
		matrix = np.zeros((2, 9), dtype=np.float32)
		# 判断庄闲大小扑谁大，根据庄闲ID编码
		# 庄家大扑大
		if compare_big_cards == "landlord":
			matrix[0][:] = 1

		# 闲家大扑大
		elif compare_big_cards == "farmer":
			matrix[1][:] = 1

		return matrix.flatten('A')

	@staticmethod
	def compare_big_res(compare_big_cards):
		"""
		todo: 闲家撒扑时，比较庄闲大扑大小
		"""
		if not compare_big_cards or compare_big_cards == "draw":
			return np.zeros(18, dtype=np.float32)
		matrix = np.zeros((2, 9), dtype=np.float32)
		# 判断庄闲大小扑谁大，根据庄闲ID编码
		# 庄家大扑大
		if compare_big_cards == "landlord":
			matrix[0][:] = 1

		# 闲家大扑大
		elif compare_big_cards == "farmer":
			matrix[1][:] = 1

		return matrix.flatten('A')

	def encode_history_actions(self, history_actions):
		"""
		todo: 编码历史对局动作序列
		"""
		if not history_actions:
			return np.zeros(9 * 3, dtype=np.float32)
		matrix = np.zeros((9, 3), dtype=np.float32)
		for idx, actions in enumerate(history_actions):
			for i, action in enumerate(actions):
				matrix[idx][i] = 1

		# 将存储的历史动作序列清零
		if len(self.game.history_actions) > 8:
			self.game.history_actions = []

		return matrix.flatten('A')

	def encode_history_round(self):
		"""
		编码历史对局
		"""
		# 对局信息记录为空时，则全部返回零
		if not self.game.history_rounds:
			return np.zeros(10 * 294, dtype=np.float32)
		# 十局历史记录编码
		collect_remain_by_matrix = np.zeros((10, 294), dtype=np.float32)
		for idx, history_round in enumerate(self.game.history_rounds):
			winner_id_round = self.encode_landlord_or_farmer(history_round["winner_role"], history_round["winner_id"] == -1)
			winner_big_cards = self.encode_big_and_small_cards(history_round["landlord_cards"][:2])
			winner_small_cards = self.encode_big_and_small_cards(history_round["landlord_cards"][2:])
			loser_big_cards = self.encode_big_and_small_cards(history_round["farmer_cards"][:2])
			loser_small_cards = self.encode_big_and_small_cards(history_round["farmer_cards"][2:])
			compare_winner_big_cards = self.compare_big_res(history_round["compare_big_cards"])
			compare_winner_small_cards = self.compare_big_res(history_round["compare_small_cards"])
			remain_by_matrix = self.encode_cards(history_round["remain_cards"])

			package_matrix = np.hstack((
				winner_id_round,
				winner_big_cards,
				winner_small_cards,
				loser_big_cards,
				loser_small_cards,
				compare_winner_big_cards,
				compare_winner_small_cards,
				remain_by_matrix,
			))

			collect_remain_by_matrix[idx] = package_matrix.flatten('A')

		if len(self.game.history_rounds) >= 10:
			self.game.history_rounds.clear()

		return collect_remain_by_matrix.flatten('A')