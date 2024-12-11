# -*- coding: utf-8 -*-


class Card:
	"""
	打妖怪卡牌类
	"""
	def __init__(self, card_type, card_value):
		"""
		初始化卡牌属性参数
		"""
		self.card_type = card_type
		self.card_value = card_value
		self.index_num = 0

	def set_index_num(self, index_num):
		"""
		索引
		"""
		self.index_num = index_num

	def get_card_type(self):
		"""
		获取卡牌类型
		"""
		return self.card_type

	def get_card_value(self):
		"""
		获取卡牌值
		"""
		return self.card_value