# -*- coding: utf-8 -*-

import numpy as np

from copy import deepcopy
from game import Game


class ConnectFourGame(Game):
	"""
	todo: 表示游戏板及其逻辑

	row: 一个整数，指示板行的长度
	column: 指示板列长度的整数.
	connect: 一个整数，指示要连接的段数
	current_player: 用于跟踪当前玩家的整数
	state: 以矩阵形式存储游戏状态的列表
	action_size: 一个整数，指示板方块的总数
	directions: 包含用于检查有效移动的元组的字典
	"""

	def __init__(self):
		"""
		todo: 使用初始板状态初始化 ConnectFourGame
		"""
		super().__init__()
		self.row = 6
		self.column = 7
		self.connect = 4
		self.current_player = 1
		self.state = []
		self.action_size = self.row * self.column

		# 创建一个n x n矩阵来表示游戏界面
		for i in range(self.row):
			self.state.append([0 * j for j in range(self.column)])

		self.state = np.array(self.state)

		# 移动方向
		self.directions = {
			0: (-1, -1),
			1: (-1, 0),
			2: (-1, 1),
			3: (0, -1),
			4: (0, 1),
			5: (1, -1),
			6: (1, 0),
			7: (1, 1)
		}

	def clone(self):
		"""
		todo: 创建游戏对象的深层克隆
		"""
		game_clone = ConnectFourGame()
		game_clone.state = deepcopy(self.state)
		game_clone.current_player = self.current_player
		return game_clone

	def play_action(self, action):
		"""
		todo: 在游戏界面进行动作
		动作:(行、列)形式的元组
		"""
		x = action[1]
		y = action[2]

		self.state[x][y] = self.current_player
		self.current_player = -self.current_player

	def get_valid_moves(self, current_player):
		"""
		todo: 返回移动及其有效性的列表

		在电路板中搜索零(0)。0表示一个空正方形
		包含以下形式的移动的列表(有效性、行、列)
		"""
		valid_moves = []

		for x in range(self.row):
			for y in range(self.column):
				if self.state[x][y] == 0:
					if x + 1 == self.row:
						valid_moves.append((1, x, y))
					elif x + 1 < self.row:
						if self.state[x + 1][y] != 0:
							valid_moves.append((1, x, y))
						else:
							valid_moves.append((0, None, None))
				else:
					valid_moves.append((0, None, None))

		return np.array(valid_moves)

	def check_game_over(self, current_player):
		"""
		todo: 检查游戏是否结束并返回可能的获胜者

		有3种可能的情况:
			1.比赛结束了，有一个赢家
			2.比赛结束了，但这是平局
			3.游戏还没有结束

		[winner: 1, loser: -1, draw: 0]
		"""

		player_a = current_player
		player_b = -current_player

		for x in range(self.row):
			for y in range(self.column):
				player_a_count = 0
				player_b_count = 0

				# 搜索玩家 A
				if self.state[x][y] == player_a:
					player_a_count += 1

					# 在所有8个方向上搜索类似的作品
					for i in range(len(self.directions)):
						d = self.directions[i]

						r = x + d[0]
						c = y + d[1]

						if r < self.row and c < self.column:
							count = 1

							# 继续搜索连接
							while True:
								r = x + d[0] * count
								c = y + d[1] * count

								count += 1

								if 0 <= r < self.row and 0 <= c < self.column:
									if self.state[r][c] == player_a:
										player_a_count += 1
									else:
										break
								else:
									break

						if player_a_count >= self.connect:
							return True, 1

						player_a_count = 1

				# 搜索玩家 B
				if self.state[x][y] == player_b:
					player_b_count += 1

					# 在所有8个方向上搜索类似的作品
					for i in range(len(self.directions)):
						d = self.directions[i]

						r = x + d[0]
						c = y + d[1]

						if r < self.row and c < self.column:
							count = 1

							# 继续搜索连接
							while True:
								r = x + d[0] * count
								c = y + d[1] * count

								count += 1

								if 0 <= r < self.row and 0 <= c < self.column:
									if self.state[r][c] == player_b:
										player_b_count += 1
									else:
										break
								else:
									break

						if player_b_count >= self.connect:
							return True, -1

						player_b_count = 1

		# 还有动作，所以游戏还没有结束
		valid_moves = self.get_valid_moves(current_player)
		print(valid_moves)

		for move in valid_moves:
			if move[0] is 1:
				return False, 0

		# 如果没有剩余的动作，游戏就结束了，没有赢家
		return True, 0

	def print_board(self):
		"""
		打印输出
		"""
		print("   0    1    2    3    4    5    6")
		for x in range(self.row):
			print(x, end='')
			for y in range(self.column):
				if self.state[x][y] == 0:
					print('  -  ', end='')
				elif self.state[x][y] == 1:
					print('  X  ', end='')
				elif self.state[x][y] == -1:
					print('  O  ', end='')
			print('\n')
		print('\n')
