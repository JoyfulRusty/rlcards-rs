# -*- coding: utf-8

from rlcards.envs import Env
from collections import OrderedDict
from rlcards.games.mahjong.utils import *
from rlcards.games.mahjong.game import MahjongGame as Game


class MahjongEnv(Env):
	"""
	麻将游戏环境
	"""
	def __init__(self, config):
		"""
		初始化麻将游戏
		"""
		self.name = 'mahjong'
		self.game = Game()
		super().__init__(config, self.game)
		self.action_id = new_card_encoding_dict
		self.state_shape = [[1024], [1024], [1024]]
		self.action_shape = [[39] for _ in range(self.num_players)]
		self.decode_action_id = {self.action_id[key]: key for key in self.action_id.keys()}

	def extract_state(self, state):
		"""
		抽取状态数据
		"""
		# 状态数据
		obs = self.get_obs(state)
		extracted_state = {
			'obs': obs,
			'z_obs': obs.reshape(11, 107).astype(np.float32),
			'actions': self._get_legal_actions(),
			'row_state': state,
			'row_legal_actions': [a for a in state['actions']],
			'action_record': self.action_recorder
		}

		# print("输出玩家出牌记录: ", len(self.action_recorder), self.action_recorder)

		return extracted_state

	@staticmethod
	def get_obs(state):
		"""
		状态数据编码
		"""
		# 碰、杠(卡牌)
		piles = state['piles']

		# 卡牌编码
		curr_hand_cards = encode_cards(state['curr_hand_cards'])
		other_hand_cards = encode_cards(state['other_hand_cards'])

		play_cards_0 = encode_cards(state['played_cards'][0])
		play_cards_1 = encode_cards(state['played_cards'][1])
		play_cards_2 = encode_cards(state['played_cards'][2])

		# 卡牌数量编码
		card_nums = encode_nums(state['hand_card_nums'])

		# 桌牌(不包含碰、杠、胡等操作)
		table_history = encode_cards(state['table'])

		# 动作历史(包含碰、杠、胡等操作)
		action_history = action_seq_history(state['action_seq_history'])

		# 对局中出现的碰、杠牌
		piles_cards = encode_cards(pile2list(piles))

		# 合法动作
		legal_actions = encode_legal_actions(state['actions'])

		# 上一位玩家出牌动作
		last_action = encode_last_action(state['action_seq_history'])

		obs = np.hstack((
			curr_hand_cards,
			other_hand_cards,
			card_nums,
			play_cards_0,
			play_cards_1,
			play_cards_2,
			table_history,
			action_history,
			piles_cards,
			legal_actions,
			last_action
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
		if action_id < 39:
			candidates = self.game.get_legal_actions()
			for card in candidates:
				if card == action:
					action = card
					break
		return action

	def _get_legal_actions(self):
		"""
		获取合法动作
		"""
		legal_action_id = {}
		# 合法动作
		legal_actions = self.game.get_legal_actions()
		if legal_actions:
			for action in legal_actions:
				legal_action_id[self.action_id[action]] = None
		else:
			print(legal_actions)
			print("##########################")
			print("No Legal Actions")
			print("judge_name: ", self.game.judge.judge_name(self.game))
			print("game_is_over: ", self.game.is_over())
			print([len(p.piles) for p in self.game.players])
			print("current_player_state: ", self.game.get_state(self.game.round.curr_player_id))

		return OrderedDict(legal_action_id)

	def get_action_feature(self, action):
		"""
		TODO: 动作特征编码
		"""
		if isinstance(action, dict):
			return np.zeros(39, dtype=np.int8)
		action = self._decode_action(action)
		matrix = np.zeros(39, dtype=np.int8)
		index = new_card_encoding_dict[action]
		matrix[index] = 1
		return matrix