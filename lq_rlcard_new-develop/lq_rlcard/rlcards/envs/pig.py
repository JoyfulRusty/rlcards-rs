# -*- coding: utf-8

import numpy as np

from rlcards.envs import Env
from collections import OrderedDict
from rlcards.const.pig.const import CARD_INDEX
from rlcards.games.newpig.game import GzGame as Game
from rlcards.games.newpig.utils import new_card_encoding_dict


class PigEnv(Env):
	"""
	麻将游戏环境
	"""
	def __init__(self, config):
		"""
		初始化麻将游戏
		"""
		self.name = 'pig'
		self.game = Game()
		super().__init__(config, self.game)
		self.action_id = new_card_encoding_dict
		self.state_shape = [[664], [664], [664], [664]]
		self.action_shape = [[54] for _ in range(self.num_players)]
		self.decode_action_id = {self.action_id[key]: key for key in self.action_id.keys()}

	def extract_state(self, state):
		"""
		抽取状态数据
		"""
		# 状态数据
		obs = self.calc_parse_obs(state)
		extracted_state = {
			'obs': obs,
			'z_obs': obs.reshape(16, 115).astype(np.float32),
			'legal_actions': self._get_legal_actions(),
			'row_state': state,
			'row_legal_actions': [a for a in state['legal_actions']],
			'action_record': self.action_recorder
		}

		# print("输出玩家出牌记录: ", len(self.action_recorder), self.action_recorder)

		return extracted_state

	def calc_parse_obs(self, state):
		"""
		编码解析对应状态数据
		"""
		if state['self'] == 0:
			return self.get_obs_down(state)
		elif state['self'] == 1:
			return self.get_obs_right(state)
		elif state['self'] == 2:
			return self.get_obs_up(state)
		return self.get_obs_left(state)

	def get_obs_down(self, state):
		"""
		todo: down
		"""
		down_hand_cards = self.encode_cards(state['curr_hand_cards'])
		down_played_cards = self.encode_cards(state['played_cards']['down'])
		down_light_cards = self.encode_cards(state['light_cards']['down'])
		down_remain_score_cards = self.encode_cards(state['remain_score_cards']['down'])
		down_receive_score_cards = self.encode_cards(state['receive_score_cards']['down'])
		down_all_receive_cards = self.encode_cards(state['all_receive_cards']['down'])

		right_hand_cards = self.encode_cards(state['other_hand_cards']['right'])
		right_played_cards = self.encode_cards(state['played_cards']['right'])
		right_light_cards = self.encode_cards(state['light_cards']['right'])
		right_remain_score_cards = self.encode_cards(state['remain_score_cards']['right'])
		right_receive_score_cards = self.encode_cards(state['receive_score_cards']['right'])
		right_all_receive_cards = self.encode_cards(state['all_receive_cards']['right'])

		up_hand_cards = self.encode_cards(state['other_hand_cards']['up'])
		up_played_cards = self.encode_cards(state['played_cards']['up'])
		up_light_cards = self.encode_cards(state['light_cards']['up'])
		up_remain_score_cards = self.encode_cards(state['remain_score_cards']['up'])
		up_receive_score_cards = self.encode_cards(state['receive_score_cards']['up'])
		up_all_receive_cards = self.encode_cards(state['all_receive_cards']['up'])

		left_hand_cards = self.encode_cards(state['other_hand_cards']['left'])
		left_played_cards = self.encode_cards(state['played_cards']['left'])
		left_light_cards = self.encode_cards(state['light_cards']['left'])
		left_remain_score_cards = self.encode_cards(state['remain_score_cards']['left'])
		left_receive_score_cards = self.encode_cards(state['receive_score_cards']['left'])
		left_all_receive_cards = self.encode_cards(state['all_receive_cards']['left'])

		last_action = self.encode_legal_actions(state['last_action'])
		max_round_action = self.encode_legal_actions(state['max_round_action'])
		collect_position = self.encode_role_position(state['collect_position'])
		round_same_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		round_other_suit_action = self.encode_legal_actions(state['round_other_suit_action'])
		turn_cards = self.encode_cards(state['turn_cards'])
		round_cards = self.encode_cards(state['round_cards'])
		turn_score_cards = self.encode_cards(state['turn_score_cards'])
		round_score_cards = self.encode_cards(state['round_score_cards'])
		round_light_cards = self.encode_cards(state['round_light_cards'])
		all_cards = self.encode_cards(state['all_cards'])

		obs = np.hstack((
			down_hand_cards,
			down_played_cards,
			down_light_cards,
			down_remain_score_cards,
			down_receive_score_cards,
			down_all_receive_cards,
			right_hand_cards,
			right_played_cards,
			right_light_cards,
			right_remain_score_cards,
			right_receive_score_cards,
			right_all_receive_cards,
			up_hand_cards,
			up_played_cards,
			up_light_cards,
			up_remain_score_cards,
			up_receive_score_cards,
			up_all_receive_cards,
			left_hand_cards,
			left_played_cards,
			left_light_cards,
			left_remain_score_cards,
			left_receive_score_cards,
			left_all_receive_cards,
			last_action,
			max_round_action,
			collect_position,
			round_same_suit_action,
			round_other_suit_action,
			turn_cards,
			round_cards,
			turn_score_cards,
			round_score_cards,
			round_light_cards,
			all_cards
		))

		return obs

	def get_obs_right(self, state):
		"""
		todo: right
		"""
		right_hand_cards = self.encode_cards(state['curr_hand_cards'])
		right_played_cards = self.encode_cards(state['played_cards']['right'])
		right_light_cards = self.encode_cards(state['light_cards']['right'])
		right_remain_score_cards = self.encode_cards(state['remain_score_cards']['right'])
		right_receive_score_cards = self.encode_cards(state['receive_score_cards']['right'])
		right_all_receive_cards = self.encode_cards(state['all_receive_cards']['right'])

		up_hand_cards = self.encode_cards(state['other_hand_cards']['up'])
		up_played_cards = self.encode_cards(state['played_cards']['up'])
		up_light_cards = self.encode_cards(state['light_cards']['up'])
		up_remain_score_cards = self.encode_cards(state['remain_score_cards']['up'])
		up_receive_score_cards = self.encode_cards(state['receive_score_cards']['up'])
		up_all_receive_cards = self.encode_cards(state['all_receive_cards']['up'])

		left_hand_cards = self.encode_cards(state['other_hand_cards']['left'])
		left_played_cards = self.encode_cards(state['played_cards']['left'])
		left_light_cards = self.encode_cards(state['light_cards']['left'])
		left_remain_score_cards = self.encode_cards(state['remain_score_cards']['left'])
		left_receive_score_cards = self.encode_cards(state['receive_score_cards']['left'])
		left_all_receive_cards = self.encode_cards(state['all_receive_cards']['left'])

		down_hand_cards = self.encode_cards(state['other_hand_cards']['down'])
		down_played_cards = self.encode_cards(state['played_cards']['down'])
		down_light_cards = self.encode_cards(state['light_cards']['down'])
		down_remain_score_cards = self.encode_cards(state['remain_score_cards']['down'])
		down_receive_score_cards = self.encode_cards(state['receive_score_cards']['down'])
		down_all_receive_cards = self.encode_cards(state['all_receive_cards']['down'])

		last_action = self.encode_legal_actions(state['last_action'])
		max_round_action = self.encode_legal_actions(state['max_round_action'])
		collect_position = self.encode_role_position(state['collect_position'])
		round_same_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		round_other_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		turn_cards = self.encode_cards(state['turn_cards'])
		round_cards = self.encode_cards(state['round_cards'])
		turn_score_cards = self.encode_cards(state['turn_score_cards'])
		round_score_cards = self.encode_cards(state['round_score_cards'])
		round_light_cards = self.encode_cards(state['round_light_cards'])
		all_cards = self.encode_cards(state['all_cards'])

		obs = np.hstack((
			right_hand_cards,
			right_played_cards,
			right_light_cards,
			right_remain_score_cards,
			right_receive_score_cards,
			right_all_receive_cards,
			up_hand_cards,
			up_played_cards,
			up_light_cards,
			up_remain_score_cards,
			up_receive_score_cards,
			up_all_receive_cards,
			left_hand_cards,
			left_played_cards,
			left_light_cards,
			left_remain_score_cards,
			left_receive_score_cards,
			left_all_receive_cards,
			down_hand_cards,
			down_played_cards,
			down_light_cards,
			down_remain_score_cards,
			down_receive_score_cards,
			down_all_receive_cards,
			last_action,
			max_round_action,
			collect_position,
			round_same_suit_action,
			round_other_suit_action,
			turn_cards,
			round_cards,
			turn_score_cards,
			round_score_cards,
			round_light_cards,
			all_cards
		))

		return obs

	def get_obs_up(self, state):
		"""
		todo: up
		"""
		up_hand_cards = self.encode_cards(state['curr_hand_cards'])
		up_played_cards = self.encode_cards(state['played_cards']['up'])
		up_light_cards = self.encode_cards(state['light_cards']['up'])
		up_remain_score_cards = self.encode_cards(state['remain_score_cards']['up'])
		up_receive_score_cards = self.encode_cards(state['receive_score_cards']['up'])
		up_all_receive_cards = self.encode_cards(state['all_receive_cards']['up'])

		left_hand_cards = self.encode_cards(state['other_hand_cards']['left'])
		left_played_cards = self.encode_cards(state['played_cards']['left'])
		left_light_cards = self.encode_cards(state['light_cards']['left'])
		left_remain_score_cards = self.encode_cards(state['remain_score_cards']['left'])
		left_receive_score_cards = self.encode_cards(state['receive_score_cards']['left'])
		left_all_receive_cards = self.encode_cards(state['all_receive_cards']['left'])

		down_hand_cards = self.encode_cards(state['other_hand_cards']['down'])
		down_played_cards = self.encode_cards(state['played_cards']['down'])
		down_light_cards = self.encode_cards(state['light_cards']['down'])
		down_remain_score_cards = self.encode_cards(state['remain_score_cards']['down'])
		down_receive_score_cards = self.encode_cards(state['receive_score_cards']['down'])
		down_all_receive_cards = self.encode_cards(state['all_receive_cards']['down'])

		right_hand_cards = self.encode_cards(state['other_hand_cards']['right'])
		right_played_cards = self.encode_cards(state['played_cards']['right'])
		right_light_cards = self.encode_cards(state['light_cards']['right'])
		right_remain_score_cards = self.encode_cards(state['remain_score_cards']['right'])
		right_receive_score_cards = self.encode_cards(state['receive_score_cards']['right'])
		right_all_receive_cards = self.encode_cards(state['all_receive_cards']['right'])

		last_action = self.encode_legal_actions(state['last_action'])
		max_round_action = self.encode_legal_actions(state['max_round_action'])
		collect_position = self.encode_role_position(state['collect_position'])
		round_same_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		round_other_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		turn_cards = self.encode_cards(state['turn_cards'])
		round_cards = self.encode_cards(state['round_cards'])
		turn_score_cards = self.encode_cards(state['turn_score_cards'])
		round_score_cards = self.encode_cards(state['round_score_cards'])
		round_light_cards = self.encode_cards(state['round_light_cards'])
		all_cards = self.encode_cards(state['all_cards'])

		obs = np.hstack((
			up_hand_cards,
			up_played_cards,
			up_light_cards,
			up_remain_score_cards,
			up_receive_score_cards,
			up_all_receive_cards,
			left_hand_cards,
			left_played_cards,
			left_light_cards,
			left_remain_score_cards,
			left_receive_score_cards,
			left_all_receive_cards,
			down_hand_cards,
			down_played_cards,
			down_light_cards,
			down_remain_score_cards,
			down_receive_score_cards,
			down_all_receive_cards,
			right_hand_cards,
			right_played_cards,
			right_light_cards,
			right_remain_score_cards,
			right_receive_score_cards,
			right_all_receive_cards,
			last_action,
			max_round_action,
			collect_position,
			round_same_suit_action,
			round_other_suit_action,
			turn_cards,
			round_cards,
			turn_score_cards,
			round_score_cards,
			round_light_cards,
			all_cards
		))

		return obs

	def get_obs_left(self, state):
		"""
		todo: left
		"""
		left_hand_cards = self.encode_cards(state['curr_hand_cards'])
		left_played_cards = self.encode_cards(state['played_cards']['left'])
		left_light_cards = self.encode_cards(state['light_cards']['left'])
		left_remain_score_cards = self.encode_cards(state['remain_score_cards']['left'])
		left_receive_score_cards = self.encode_cards(state['receive_score_cards']['left'])
		left_all_receive_cards = self.encode_cards(state['all_receive_cards']['left'])

		down_hand_cards = self.encode_cards(state['other_hand_cards']['down'])
		down_played_cards = self.encode_cards(state['played_cards']['down'])
		down_light_cards = self.encode_cards(state['light_cards']['down'])
		down_remain_score_cards = self.encode_cards(state['remain_score_cards']['down'])
		down_receive_score_cards = self.encode_cards(state['receive_score_cards']['down'])
		down_all_receive_cards = self.encode_cards(state['all_receive_cards']['down'])

		right_hand_cards = self.encode_cards(state['other_hand_cards']['right'])
		right_played_cards = self.encode_cards(state['played_cards']['right'])
		right_light_cards = self.encode_cards(state['light_cards']['right'])
		right_remain_score_cards = self.encode_cards(state['remain_score_cards']['right'])
		right_receive_score_cards = self.encode_cards(state['receive_score_cards']['right'])
		right_all_receive_cards = self.encode_cards(state['all_receive_cards']['right'])

		up_hand_cards = self.encode_cards(state['other_hand_cards']['up'])
		up_played_cards = self.encode_cards(state['played_cards']['up'])
		up_light_cards = self.encode_cards(state['light_cards']['up'])
		up_remain_score_cards = self.encode_cards(state['remain_score_cards']['up'])
		up_receive_score_cards = self.encode_cards(state['receive_score_cards']['up'])
		up_all_receive_cards = self.encode_cards(state['all_receive_cards']['up'])

		last_action = self.encode_legal_actions(state['last_action'])
		max_round_action = self.encode_legal_actions(state['max_round_action'])
		collect_position = self.encode_role_position(state['collect_position'])
		round_same_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		round_other_suit_action = self.encode_legal_actions(state['round_same_suit_action'])
		turn_cards = self.encode_cards(state['turn_cards'])
		round_cards = self.encode_cards(state['round_cards'])
		turn_score_cards = self.encode_cards(state['turn_score_cards'])
		round_score_cards = self.encode_cards(state['round_score_cards'])
		round_light_cards = self.encode_cards(state['round_light_cards'])
		all_cards = self.encode_cards(state['all_cards'])

		obs = np.hstack((
			left_hand_cards,
			left_played_cards,
			left_light_cards,
			left_remain_score_cards,
			left_receive_score_cards,
			left_all_receive_cards,
			down_hand_cards,
			down_played_cards,
			down_light_cards,
			down_remain_score_cards,
			down_receive_score_cards,
			down_all_receive_cards,
			right_hand_cards,
			right_played_cards,
			right_light_cards,
			right_remain_score_cards,
			right_receive_score_cards,
			right_all_receive_cards,
			up_hand_cards,
			up_played_cards,
			up_light_cards,
			up_remain_score_cards,
			up_receive_score_cards,
			up_all_receive_cards,
			last_action,
			max_round_action,
			collect_position,
			round_same_suit_action,
			round_other_suit_action,
			turn_cards,
			round_cards,
			turn_score_cards,
			round_score_cards,
			round_light_cards,
			all_cards
		))

		return obs

	def get_payoffs(self):
		"""
		TODO: 每位玩家收益列表
		"""
		return self.game.judge.judge_payoffs(self.game)

	def _decode_action(self, action_id):
		"""
		解码动作
		"""
		# 解码动作
		action = self.decode_action_id[action_id]
		if action_id < 55:
			candidates = self.game.get_legal_actions()
			for card in candidates:
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

	def get_action_feature(self, action):
		"""
		TODO: 动作特征编码
		"""
		if isinstance(action, dict):
			return np.zeros(54, dtype=np.int8)
		action = self._decode_action(action)
		matrix = np.zeros(54, dtype=np.int8)
		index = new_card_encoding_dict[action]
		matrix[index] = 1
		return matrix

	@staticmethod
	def encode_role_position(role=''):
		"""
		编码上一位玩家位置
		"""
		if not role:
			return np.zeros(4, dtype=np.float32)
		matrix = np.zeros(4, dtype=np.float32)
		if role == 'down':
			matrix[0] = 1
		elif role == 'right':
			matrix[1] = 1
		elif role == 'up':
			matrix[2] = 1
		elif role == 'left':
			matrix[3] = 1
		return matrix

	@staticmethod
	def encode_legal_actions(legal_actions):
		"""
		编码合法动作
		"""
		if not legal_actions:
			return np.zeros(54, dtype=np.float32)
		matrix = np.zeros([4, 13], dtype=np.float32)
		ghost_matrix = np.zeros(2, dtype=np.float32)
		ghost_count = 0
		for card in legal_actions:
			card_type = (card // 100) - 1
			if card_type == 4:
				ghost_count += 1
				continue
			matrix[card_type][CARD_INDEX[card % 100]] = 1
		if ghost_count > 0:
			ghost_matrix[:ghost_count] = 1
		return np.concatenate((matrix.flatten('A'), ghost_matrix))

	@staticmethod
	def encode_cards(score_cards):
		"""
		编码分牌
		"""
		if not score_cards:
			return np.zeros(54, dtype=np.float32)
		matrix = np.zeros([4, 13], dtype=np.float32)
		ghost_matrix = np.zeros(2, dtype=np.float32)
		ghost_count = 0
		for card in score_cards:
			card_type = (card // 100) - 1
			if card_type == 4:
				ghost_count += 1
				continue
			matrix[card_type][CARD_INDEX[card % 100]] = 1
		if ghost_count > 1:
			ghost_matrix[:ghost_count] = 1

		return np.concatenate((matrix.flatten('A'), ghost_matrix))