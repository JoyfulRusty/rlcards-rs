# -*- coding: utf-8 -*-

from rlcards.const.doudizhu import const as CONST

class Card:
	"""
	卡牌: 单张卡牌的花色和点数
	note:
		标准卡牌的花色类型: [S, H, D, C, BJ, RJ] 之一，意思是 [黑桃、红心、方块、梅花、黑小丑、红小丑]
		卡牌变量排名: [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K] 之一
	"""
	suit = None
	rank = None
	valid_suit = CONST.CARD_SUIT
	valid_rank = CONST.CARD_RANK

	def __init__(self, suit, rank):
		"""
		初始化卡牌的花色类型和等级
		:param suit: 卡牌的花色类型
		:param rank: 卡牌的等级
		"""
		self.suit = suit
		self.rank = rank

	def __eq__(self, other):
		"""
		比较isinstance
		:param other: 其他牌
		:return: 是否为Card
		"""
		if isinstance(other, Card):
			return self.rank == other.rank and self.suit == other.suit
		else:
			# 不与不想关的类型进行比较
			return NotImplemented

	def __hash__(self):
		"""
		卡牌hash
		"""
		suit_index = Card.valid_suit.index(self.suit)
		rank_index = Card.valid_rank.index(self.rank)
		return rank_index + 100 * suit_index

	def __str__(self):
		"""
		打印卡牌的字符串形式
		:return: string: 一张牌的等级和花色的组合。例如：AS, 5H, JD, 3C, ...
		"""
		return self.rank + self.suit

	def get_index(self):
		"""
		获取卡牌索引
		:return: string: 花色和牌的点数的组合。例如：1S、2H、AD、BJ、RJ...
		"""
		return self.suit + self.rank