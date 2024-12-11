# -*- coding: utf-8 -*-

import random

from rlcards.games.sytx.rules import Rules
from rlcards.games.sytx.utils import init_deck
from rlcards.games.sytx.allocat import get_better_combine


class SyDealer:
	"""
	水鱼发牌器对象
	"""
	def __init__(self, np_random):
		"""
		初始化水鱼发牌器属性参数
		"""
		self.np_random = np_random
		self.deck = init_deck()
		self.table = []
		self.left_count = 55
		self.cards_count = 4

		self.shuffle()  # 洗牌

	def shuffle(self):
		"""
		洗牌
		"""
		self.np_random.shuffle(self.deck)

	def deal_cards(self, player):
		"""
		todo: 水鱼发牌
			1.每位玩家发四张卡牌
			2.one vs one (player)
		"""
		curr_hand_cards = []
		# 每位玩家发四张卡牌
		for _ in range(self.cards_count):
			self.left_count -= 1  # 记录卡牌数量
			card = self.deck.pop(0)
			curr_hand_cards.append(card.get_card_value())

		# 获取卡牌能够组成的较好组合
		get_combine_cards = get_better_combine(curr_hand_cards)
		sort_big_small_cards = Rules.sort_big_small(get_combine_cards)

		# 更新玩家当前手牌
		player.update_cards(sort_big_small_cards)

	def set_regular_roles(self, players):
		"""
		设置固定角色
		"""
		players[0].role = "landlord"  # 庄家
		players[1].role = "farmer"  # 闲家

	def set_landlord(self, players):
		"""
		选择庄家
		"""
		sum_dice = {}
		record_dice = 0
		for player in players:
			# 摇两次骰子并相加点数
			dict_count = self.shake_dice()
			sum_dice[player.player_id] = dict_count
			# 筛选出骰子最大值进行比较
			if dict_count == record_dice:
				return player.player_id
			if dict_count > record_dice:
				record_dice = dict_count

		# 选出最大骰子数的玩家ID(one vs one)
		max_dice = [p for p, dice in sum_dice.items() if dice == record_dice][-1]

		return max_dice

	@staticmethod
	def shake_dice():
		"""
		摇骰子比点数大小
		"""
		# 计算两个骰子相加的点数大小
		dice_1 = random.randint(1, 6)
		dice_2 = random.randint(1, 6)

		return dice_1 + dice_2