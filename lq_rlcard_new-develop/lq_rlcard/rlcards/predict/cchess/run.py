# -*- coding: utf-8 -*-

import time
import random

from rlcards.utils.utils import cal_time
from rlcards.games.aichess.rlzero.config import CONFIG
from rlcards.games.aichess.rlzero.env.game import Game
from rlcards.games.aichess.rlzero.env.board import Board
from rlcards.games.aichess.rlzero.env.player import MCTSPlayer
from rlcards.games.aichess.rlzero.env.ucci import get_cep_move_func, get_cep_move_func_pika
from rlcards.games.aichess.rlzero.util.util import move_id2move_action, move_action2move_id


class ChessAgent:
	"""
	todo: 象棋AI
	"""
	def __init__(self, use_cep=True):
		"""
		初始化参数
		"""
		self.board = Board()
		self.game = Game(self.board)
		self.use_cep = use_cep

	def step(self, to_fen=None):
		"""
		计算动作
		"""
		move = None
		self.board.init_board()
		# todo: 使用象棋引擎(象眼)
		if self.use_cep:
			print("象棋引擎[Chess Engine Protocol -> U C C I]计算机器人动作")
			acts = self.board.legal_moves
			if len(acts) == 1:
				move = acts[0]
			else:
				move, _ = get_cep_move_func_pika(self.board, self.update_curr_color(1))
				if not move:
					print("当前动作为None")
					move = random.choice(self.board.legal_moves)
		# 解析计算处的动作
		actions = move_id2move_action[move]
		start_y, start_x = int(actions[0]), int(actions[1])
		end_y, end_x = int(actions[2]), int(actions[3])

		# 返回初始坐标(start_y, start_x), 移动坐标(end_y, end_x)
		return (start_x, start_y), (end_x, end_y)

	@staticmethod
	def update_curr_color(seat_id):
		"""
		更新当前玩家花色
		"""
		return {
			1: '红',
			2: '黑'
		}.get(seat_id, "红")

agent = ChessAgent()

def main():
	while True:
		start_time = time.time()
		start_position, end_position = agent.step()
		print(f"起始位置: {start_position}, 结束位置: {end_position}")
		end_time = time.time()
		print("输出计算耗时: ", end_time - start_time)
		print()

if __name__ == "__main__":
	main()