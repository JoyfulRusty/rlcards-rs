# -*- coding: utf-8 -*-


class SyPlayer:
	"""
	水鱼玩家
	"""
	def __init__(self, player_id, np_random):
		"""
		初始化水鱼玩家参数
		"""
		self.player_id = player_id
		self.np_random = np_random
		self.is_winner = False  # 是否为赢家
		self.curr_hand_cards = []  # 手牌
		self.dark_cards = []
		self.bright_cards = []
		self.combine_cards = []  # 三张明牌分扑
		self.action_reward = 0.0  # 动作奖励
		self.extra_reward = 0.0  # 额外奖励
		self.cards_reward = 0.0  # 卡牌奖励
		self.role = ''  # 玩家角色

	def update_cards(self, cards):
		"""
		更新玩家当前手牌
		"""
		self.curr_hand_cards = cards

	def receive_combine_by_3(self, combine_bright_cards):
		"""
		庄家开始喊撒扑后，庄闲开始分扑明牌
		"""
		self.combine_cards = combine_bright_cards