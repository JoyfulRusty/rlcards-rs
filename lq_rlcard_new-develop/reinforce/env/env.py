# -*- coding: utf-8 -*-

from reinforce.env import seeding
from typing import Dict, Tuple, List

class Env:
	"""
	棋牌游戏环境基类
	"""

	def __init__(self, config: Dict, game) -> None:
		"""
		初始化棋牌游戏环境基类属性参数
		"""
		self.game = game
		self.agents = []
		self.time_step = 0
		self.np_random = 0
		self.game_np_random = 0
		self.action_recorder = []
		self.seed(config["seed"])
		self.num_players = self.game.get_num_players()
		self.num_actions = self.game.get_num_actions()
		self.allow_step_back = self.game.get_allow_step_back = config["allow_step_back"]

	def reset(self) -> Tuple:
		"""
		重置环境
		"""
		state, player_id = self.game.init_game()
		self.action_recorder = []
		return self.extract_state(state), player_id

	def step(self, action: int, row_action: bool = False) -> Tuple:
		"""
		更新
		"""
		if not row_action:
			action = self._decode_action(action)
		self.time_step += 1
		self.action_recorder.append((self.get_player_id(), action))
		next_state, player_id = self.game.step(action)
		return self.extract_state(next_state), player_id

	def set_agents(self, agent) -> None:
		"""
		设置智能体代理
		"""
		self.agents = agent

	def run(self, is_training: bool = False) -> Tuple:
		"""
		todo: 训练与评估
		"""
		trajectories = [[] for _ in range(self.num_players)]
		state, player_id = self.reset()
		trajectories[player_id].append(state)
		while not self.is_over():
			if not is_training:
				action, _ = self.agents[player_id].eval_step(state)
			else:
				action = self.agents[player_id].step(state)

			trajectories[player_id].append(action)
			next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
			state = next_state
			player_id = next_player_id
			if not self.game.is_over():
				trajectories[player_id].append(state)

		for player_id in range(self.num_players):
			state = self.get_state(player_id)
			trajectories[player_id].append(state)

		# 模型奖励参数
		payoffs = self.get_payoffs()

		return trajectories, payoffs, self.num_players, player_id

	def step_back(self) -> Tuple:
		"""
		动作序列反向迭代
		"""
		if not self.allow_step_back:
			raise Exception('Step back is off. To use step_back, please set allow_step_back=True in env.make')
		if not self.game.step_back():
			return ()
		self.game.step_back()
		player_id = self.get_player_id()
		state = self.get_state(player_id)
		return state, player_id

	def is_over(self) -> bool:
		"""
		游戏是否结束
		"""
		return self.game.is_over()

	def get_player_id(self) -> int:
		"""
		获取当前玩家id
		"""
		return self.game.get_player_id()

	def get_state(self, player_id: int) -> Dict:
		"""
		获取当前玩家状态参数
		"""
		return self.extract_state(self.game.get_state(player_id))

	def seed(self, seed: int = None) -> None:
		"""
		设置随机种子
		"""
		self.np_random, seed = seeding.np_random(seed)
		self.game_np_random = self.np_random

	def get_payoffs(self) -> List:
		"""
		获取所有玩家的奖励
		"""
		raise NotImplementedError

	def extract_state(self, state: dict) -> Dict:
		"""
		提取状态参数
		"""
		raise NotImplementedError

	def _decode_action(self, action_id: int) -> str:
		"""
		解码动作
		"""
		raise NotImplementedError

	def _get_legal_actions(self) -> List:
		"""
		获取当前玩家合法动作
		"""
		raise NotImplementedError