# -*- coding: utf-8 -*-

import time
import numpy as np

from rlcards.games.aichess.xqengine.search import Search
from rlcards.games.aichess.xqengine.const import X_POS_STR
from rlcards.games.aichess.xqengine.position import Position
from rlcards.games.aichess.xqengine.chess import move2icc, icc2move


class ChessAgent:
	"""
	构建象棋计算代理
	"""
	def __init__(self):
		"""
		初始化参数
		"""
		self.depth = 64
		self.hash_level = 16
		self.search_time_ms = 1000

	def step(self, last_move=None, fen=''):
		"""
		解析移动move
		"""
		pos = Position()
		if not fen:
			fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
		# 解析棋盘布局
		pos.from_fen(fen)
		search = Search(pos, self.hash_level)
		# todo: 首出
		if not last_move:
			move = search.search_main(self.depth, self.search_time_ms)
			start_move, end_move = self.parse_move(move2icc(move))
			print(f"原始坐标: {start_move}, 目标坐标: {end_move}")
			return start_move, end_move

		# todo: 非首出
		move = search.search_main(self.depth, self.search_time_ms)
		start_move, end_move = self.parse_move(move2icc(move))
		print(f"原始坐标: {start_move}, 目标坐标: {end_move}")
		return start_move, end_move

	def parse_move(self, move):
		"""
		解析移动坐标
		"""
		start_pos = self.update_x_pos(move.split('-')[0])
		end_pos = self.update_x_pos(move.split('-')[1])
		return [int(start_pos[0]), int(start_pos[1])], [int(end_pos[0]), int(end_pos[1])]

	@staticmethod
	def update_x_pos(pos):
		"""
		更新映射横坐标
		"""
		return pos.replace(pos[0], str(X_POS_STR.get(pos[0])))

agent = ChessAgent()
while True:
	agent.step()