# -*- coding: utf-8 -*-

class Chess:
	"""
	棋
	"""
	def __init__(self, belong, chess_type):
		"""
		初始化参数
		"""
		self.belong = belong  # 红/黑[0/1/-1]，-1表示位置chess_type全部都为0
		self.chess_type = chess_type  # 红棋/黑棋

	def can_move(self, to_x, to_y):
		"""
		返回是否能向此方向走
		"""
		return False