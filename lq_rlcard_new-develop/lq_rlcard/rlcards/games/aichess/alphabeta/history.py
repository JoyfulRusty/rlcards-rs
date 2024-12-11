# -*- coding: utf-8 -*-

import numpy as np

class HistoryCache:
	"""
	历史启发算法
	"""
	def __init__(self):
		"""
		初始化缓存表
		"""
		self.cache_table = np.zeros((2, 90, 90))

	def get_history_score(self, who, step):
		"""
		获取历史分数
		"""
		# 获取缓存表
		return self.cache_table[who, step.from_x * 9 + step.from_y, step.to_x * 9 + step.to_y]

	def add_history_score(self, who, step, depth):
		"""
		添加历史对局分数及深度
		"""
		# 添加缓存表
		self.cache_table[who, step.from_x * 9 + step.from_y, step.to_x * 9 + step.to_y] += 2 << depth