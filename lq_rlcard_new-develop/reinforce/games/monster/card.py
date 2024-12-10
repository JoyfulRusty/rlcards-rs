# -*- coding: utf-8 -*-


class Card:
	"""
	卡牌
	"""

	def __init__(self, card_type, card_value):
		"""
		初始化卡牌属性参数
		"""
		self.index_num = 0
		self.card_type = card_type
		self.card_value = card_value

	def set_index_num(self, index_num):
		"""
		构建卡牌索引
		"""
		self.index_num = index_num