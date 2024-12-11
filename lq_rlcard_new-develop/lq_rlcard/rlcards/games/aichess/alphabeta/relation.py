# -*- coding: utf-8 -*-

class Relation:
	"""
	关系
	"""
	def __init__(self):
		self.chess_type = 0  # 象棋类型
		self.num_attack = 0  # 吃棋次数
		self.num_guard = 0  # 保护
		self.num_attacked = 0  # 被吃棋次数
		self.num_guarded = 0  # 被保护次数
		self.attack = [0, 0, 0, 0, 0, 0]  # 吃棋
		self.attacked = [0, 0, 0, 0, 0, 0]  # 吃棋了
		self.guard = [0, 0, 0, 0, 0, 0]  # 保护
		self.guarded = [0, 0, 0, 0, 0, 0]  # 保护了