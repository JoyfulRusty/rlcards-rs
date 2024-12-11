# -*- coding: utf-8 -*-

import numpy as np

from rlcards.envs.env import Env
from collections import Counter, OrderedDict
from rlcards.games.monster.game import MonsterGame as Game
from rlcards.games.monster.utils import new_card_encoding_dict
from rlcards.const.monster.const import ActionType, Card2Column, NumOnes2Array

class MonsterEnv(Env):
	"""
	打妖怪游戏环境
	"""
	def __init__(self, config):
		"""
		初始化打妖怪属性参数
		"""
		self.name = 'monster'
		self.game = Game()
		super().__init__(config, self.game)

		self.action_id = new_card_encoding_dict
		self.state_shape = [[664], [664], [664], [664]]  # lstm -> [201], [201], [201], [201]
		self.action_shape = [[32] for _ in range(self.num_players)]
		self.decode_action_id = {self.action_id[key]: key for key in self.action_id.keys()}

	def extract_state(self, state):
		"""
		抽取状态编码
		"""
		obs, action_history = self.calc_parse_obs(state)
		encode_state = {
			'obs': obs,
			'legal_actions': self._get_legal_actions(),
			'raw_obs': state,
			'z_obs': obs.reshape(4, 95).astype(np.float32),
			'raw_legal_actions': [a for a in state['actions']],
			'action_record': self.action_recorder,
			'action_history': action_history
		}

		# print("输出出牌记录: ", self.action_recorder)

		return encode_state

	def calc_parse_obs(self, state):
		"""
		选中玩家对应的obs计算流程
		"""
		if state['self'] == 0:
			return self.get_obs_down(state)
		elif state['self'] == 1:
			return self.get_obs_right(state)
		elif state['self'] == 2:
			return self.get_obs_up(state)
		return self.get_obs_left(state)

	@staticmethod
	def get_obs(state):
		"""
		状态数据编码
		"""
		curr_hand_cards = encode_cards(state['curr_hand_cards'])
		other_hand_cards = encode_other_cards(state['other_hand_cards'], state['bust'])
		played_cards = encode_played_cards(state['played_cards'], False)
		hand_card_nums = encode_nums(state['hand_card_nums'], state['bust'])
		legal_actions = encode_legal_actions(state['actions'])
		last_action = encode_last_action(state['traces'])
		action_history = action_seq_history(state['traces'])

		obs = np.hstack((
			curr_hand_cards,
			other_hand_cards,
			played_cards,
			hand_card_nums,
			legal_actions,
			last_action,
			action_history
		))

		return obs, action_history

	def get_obs_down(self, state):
		"""
		下家[0]
		"""
		# 下家手牌
		down_curr_hand_cards = encode_cards(state['curr_hand_cards'])
		down_played_cards = encode_played_cards(state['played_cards']['down'], state['bust']['down'])

		right_hand_cards = encode_other_hand_cards(state['other_hand_cards']['right'], state['bust']['right'])
		right_played_cards = encode_played_cards(state['played_cards']['right'], state['bust']['right'])
		right_bust = encode_bust(state['bust']['right'])

		up_hand_cards = encode_other_hand_cards(state['other_hand_cards']['up'], state['bust']['up'])
		up_played_cards = encode_played_cards(state['played_cards']['up'], state['bust']['up'])
		up_bust = encode_bust(state['bust']['up'])

		left_hand_cards = encode_other_hand_cards(state['other_hand_cards']['left'], state['bust']['left'])
		left_player_cards = encode_played_cards(state['played_cards']['left'], state['bust']['left'])
		left_bust = encode_bust(state['bust']['left'])

		all_cards = encode_all_cards(state['all_cards'])

		legal_actions = encode_legal_actions(state['actions'])
		last_action = encode_last_action(state['traces'])
		before_pick_action = encode_before_pick_action(state['traces'])
		action_history = action_seq_history(state['traces'])

		obs = np.hstack((
			down_curr_hand_cards,
			down_played_cards,
			right_hand_cards,
			right_played_cards,
			right_bust,
			up_hand_cards,
			up_played_cards,
			up_bust,
			left_hand_cards,
			left_player_cards,
			left_bust,
			all_cards,
			legal_actions,
			last_action,
			before_pick_action,
			action_history
		))

		return obs, action_history

	def get_obs_right(self, state):
		"""
		右家[1]
		"""
		right_curr_hand_cards = encode_cards(state['curr_hand_cards'])
		right_played_cards = encode_played_cards(state['played_cards']['right'], state['bust']['right'])

		up_hand_cards = encode_other_hand_cards(state['other_hand_cards']['up'], state['bust']['up'])
		up_played_cards = encode_played_cards(state['played_cards']['up'], state['bust']['up'])
		up_bust = encode_bust(state['bust']['up'])

		left_hand_cards = encode_other_hand_cards(state['other_hand_cards']['left'], state['bust']['left'])
		left_played_cards = encode_played_cards(state['played_cards']['left'], state['bust']['left'])
		left_bust = encode_bust(state['bust']['left'])

		down_hand_cards = encode_other_hand_cards(state['other_hand_cards']['down'], state['bust']['down'])
		down_played_cards = encode_played_cards(state['played_cards']['down'], state['bust']['down'])
		down_bust = encode_bust(state['bust']['down'])

		all_cards = encode_all_cards(state['all_cards'])

		legal_actions = encode_legal_actions(state['actions'])
		last_action = encode_last_action(state['traces'])
		before_pick_action = encode_before_pick_action(state['traces'])
		action_history = action_seq_history(state['traces'])

		obs = np.hstack((
			right_curr_hand_cards,
			right_played_cards,
			up_hand_cards,
			up_played_cards,
			up_bust,
			left_hand_cards,
			left_played_cards,
			left_bust,
			down_hand_cards,
			down_played_cards,
			down_bust,
			all_cards,
			legal_actions,
			last_action,
			before_pick_action,
			action_history
		))

		return obs, action_history

	def get_obs_up(self, state):
		"""
		上家[2]
		"""
		up_curr_hand_cards = encode_cards(state['curr_hand_cards'])
		up_played_cards = encode_played_cards(state['played_cards']['up'], state['bust']['up'])

		left_hand_cards = encode_other_hand_cards(state['other_hand_cards']['left'], state['bust']['left'])
		left_played_cards = encode_played_cards(state['played_cards']['left'], state['bust']['left'])
		left_bust = encode_bust(state['bust']['left'])

		down_hand_cards = encode_other_hand_cards(state['other_hand_cards']['down'], state['bust']['down'])
		down_played_cards = encode_played_cards(state['played_cards']['down'], state['bust']['down'])
		down_bust = encode_bust(state['bust']['down'])

		right_hand_cards = encode_other_hand_cards(state['other_hand_cards']['right'], state['bust']['right'])
		right_played_cards = encode_played_cards(state['played_cards']['right'], state['bust']['right'])
		right_bust = encode_bust(state['bust']['right'])

		all_cards = encode_all_cards(state['all_cards'])

		legal_actions = encode_legal_actions(state['actions'])
		last_action = encode_last_action(state['traces'])
		before_pick_action = encode_before_pick_action(state['traces'])
		action_history = action_seq_history(state['traces'])

		obs = np.hstack((
			up_curr_hand_cards,
			up_played_cards,
			left_hand_cards,
			left_played_cards,
			left_bust,
			down_hand_cards,
			down_played_cards,
			down_bust,
			right_hand_cards,
			right_played_cards,
			right_bust,
			all_cards,
			legal_actions,
			last_action,
			before_pick_action,
			action_history
		))

		return obs, action_history

	def get_obs_left(self, state):
		"""
		左家[3]
		"""
		left_curr_hand_cards = encode_cards(state['curr_hand_cards'])
		left_played_cards = encode_played_cards(state['played_cards']['left'], state['bust']['left'])

		down_hand_cards = encode_other_hand_cards(state['other_hand_cards']['down'], state['bust']['down'])
		down_played_cards = encode_played_cards(state['played_cards']['down'], state['bust']['down'])
		down_bust = encode_bust(state['bust']['down'])

		right_hand_cards = encode_other_hand_cards(state['other_hand_cards']['right'], state['bust']['right'])
		right_played_cards = encode_played_cards(state['played_cards']['right'], state['bust']['right'])
		right_bust = encode_bust(state['bust']['right'])

		up_hand_cards = encode_other_hand_cards(state['other_hand_cards']['up'], state['bust']['up'])
		up_played_cards = encode_played_cards(state['played_cards']['up'], state['bust']['up'])
		up_bust = encode_bust(state['bust']['up'])

		all_cards = encode_all_cards(state['all_cards'])

		legal_actions = encode_legal_actions(state['actions'])
		last_action = encode_last_action(state['traces'])
		before_pick_action = encode_before_pick_action(state['traces'])
		action_history = action_seq_history(state['traces'])

		obs = np.hstack((
			left_curr_hand_cards,
			left_played_cards,
			down_hand_cards,
			down_played_cards,
			down_bust,
			right_hand_cards,
			right_played_cards,
			right_bust,
			up_hand_cards,
			up_played_cards,
			up_bust,
			all_cards,
			legal_actions,
			last_action,
			before_pick_action,
			action_history
		))

		return obs, action_history

	def get_payoffs(self):
		"""
		对局奖励
		"""
		return self.game.judge.judge_payoffs(self.game)

	def _decode_action(self, action_id):
		"""
		解码动作
		"""
		if isinstance(action_id, dict):
			return None

		action = self.decode_action_id[action_id]
		if action_id < 30:
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

	def get_action_feature(self, action_id):
		"""
		获取动作特征
		"""
		action = self._decode_action(action_id)
		if not action or action is None:
			return np.zeros(32, dtype=np.float32)
		matrix = np.zeros(32, dtype=np.float32)
		if isinstance(action, int):
			matrix[Card2Column[action % 100]] = 1
		else:
			matrix[Card2Column[action]] = 1
		return matrix.flatten('A')

def encode_cards(cards):
	"""
	卡牌编码
	1: 'D',  # 方块♦
    2: 'C',  # 梅花♣
    3: 'H',  # 红心♥
    4: 'S',  # 黑桃♠
	"""
	if not cards:
		return np.zeros(32, dtype=np.float32)
	matrix = np.zeros((4, 7), dtype=np.float32)
	magic_matrix = np.zeros(4, dtype=np.float32)
	count_magic = 0
	for card in cards:
		idx = (card // 100) - 1
		if idx == 4:
			count_magic += 1
			continue
		matrix[idx][Card2Column[card % 100]] = 1
	if count_magic > 0:
		magic_matrix[:count_magic] = 1
	return np.concatenate((matrix.flatten('A'), magic_matrix))

def encode_other_cards(other_cards, bust):
	"""
	编码其他玩家卡牌，破产玩家卡牌编码为0
	"""
	if not other_cards:
		return np.zeros(32, dtype=np.float32)
	matrix = np.zeros((3, 8), dtype=np.float32)
	for idx, cards in enumerate(other_cards):
		cards_100 = [card % 100 for card in cards]
		cards_dict = Counter(cards_100)
		for card, nums in cards_dict.items():
			if not bust[idx]:
				matrix[idx][Card2Column[card]] = 1
			else:
				matrix[idx][Card2Column[card]] = 0

	matrix = matrix.flatten('F')

	return matrix

def encode_other_hand_cards(other_cards, bust):
	"""
	编码其他玩家手牌
	"""
	if not other_cards or bust:
		return np.zeros(32, dtype=np.float32)
	matrix = np.zeros((4, 7), dtype=np.float32)
	magic_matrix = np.zeros(4, dtype=np.float32)
	count_magic = 0
	for card in other_cards:
		idx = (card // 100) - 1
		if idx == 4:
			count_magic += 1
			continue
		matrix[idx][Card2Column[card % 100]] = 1
	if count_magic > 0:
		magic_matrix[:count_magic] = 1
	return np.concatenate((matrix.flatten('A'), magic_matrix))

def encode_bust(bust):
	"""
	编码破产玩家
	"""
	if not bust:
		# 未破产为全1
		return np.ones(1, dtype=np.float32)
	# 破产为全0
	return np.zeros(1, dtype=np.float32)

def encode_nums(hand_card_nums, bust):
	"""
	卡牌数量编码，破产将不再进行编码，全部置为0
	"""
	matrix = np.zeros((4, 8), dtype=np.float32)
	for idx, nums in enumerate(hand_card_nums):
		if not bust[idx]:
			matrix[idx][nums - 1] = 1
		else:
			matrix[idx][nums - 1] = 0

	return matrix.flatten('A')

def encode_before_pick_action(traces):
	"""
	编码导致捡牌的卡牌
	"""
	if len(traces) < 2:
		return np.zeros(8, dtype=np.float32)
	if traces[-1][1] == ActionType.PICK_CARDS:
		matrix = np.zeros(8, dtype=np.float32)
		# 编码导致捡牌的卡牌
		matrix[Card2Column[traces[-2][1] % 100]] = 1
		return matrix.flatten('A')
	return np.zeros(8, dtype=np.float32)

def encode_last_action(traces):
	"""
	编码上一个动作
	"""
	if not traces:
		return np.zeros(9, dtype=np.float32)
	matrix = np.zeros((1, 9), dtype=np.float32)
	if traces[-1][1] == ActionType.PICK_CARDS:
		matrix[:, Card2Column[traces[-1][1]]] = 1
	else:
		matrix[:, Card2Column[traces[-1][1] % 100]] = 1
	return matrix.flatten('A')

def encode_legal_actions(actions):
	"""
	编码合法动作
	"""
	if not actions:
		return np.zeros(36, dtype=np.float32)
	matrix = np.zeros((4, 9), dtype=np.float32)
	actions_dict = Counter(actions)
	for action, nums in actions_dict.items():
		if action == ActionType.PICK_CARDS:
			matrix[:, Card2Column[action]] = NumOnes2Array[nums]
		else:
			matrix[:, Card2Column[action % 100]] = NumOnes2Array[nums]

	return matrix.flatten('A')

def encode_played_cards(played_cards, bust):
	"""
	编码玩家打出的牌
	"""
	if not played_cards or bust:
		return np.zeros(32, dtype=np.float32)
	matrix = np.zeros((4, 7), dtype=np.float32)
	magic_matrix = np.zeros(4, dtype=np.float32)
	count_magic = 0
	for card in played_cards:
		idx = (card // 100) - 1
		if idx == 4:
			count_magic += 1
			continue
		matrix[idx][Card2Column[card % 100]] = 1
	if count_magic > 0:
		magic_matrix[:count_magic] = 1
	return np.concatenate((matrix.flatten('A'), magic_matrix))

def encode_all_cards(all_cards):
	"""
	编码所有卡牌
	"""
	if not all_cards:
		return np.zeros(32, dtype=np.float32)
	matrix = np.zeros((4, 7), dtype=np.float32)
	magic_matrix = np.zeros(4, dtype=np.float32)
	count_magic = 0
	for card in all_cards:
		idx = (card // 100) - 1
		if idx == 4:
			count_magic += 1
			continue
		matrix[idx][Card2Column[card % 100]] = 1
	if count_magic > 0:
		magic_matrix[:count_magic] = 1
	return np.concatenate((matrix.flatten('A'), magic_matrix))

def action_seq_history(action_seqs):
	"""
	TODO: 玩家打牌动作序列
	只对玩家动作序列的后四个动作进行编码
	以四个动作为一个滑动窗口移动编码
	"""
	if not action_seqs:
		return np.zeros(36, dtype=np.float32)
	matrix = np.zeros((4, 9), dtype=np.float32)
	for cards in action_seqs[-4:]:
		if cards[1] == ActionType.PICK_CARDS:
			matrix[cards[0]][Card2Column[cards[1]]] = 1
		else:
			matrix[cards[0]][Card2Column[cards[1] % 100]] = 1

	return matrix.flatten('A')