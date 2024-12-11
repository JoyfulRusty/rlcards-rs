# -*- coding: utf-8 -*-


class Pieces:
	"""
	todo: 构建不同的棋子
	"""
	def __init__(self, player, x, y):
		"""
		初始化参数
		"""
		self.x = x
		self.y = y
		self.player = player

	@staticmethod
	def can_move(arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		pass

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		return None

class Rooks(Pieces):
	"""
	棋: 车
	"""
	def __init__(self, player, x, y):
		"""
		初始化参数
		"""
		self.player = player
		super().__init__(player, x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		# 判断向x轴方向移动
		if self.x == moveto_x:
			step = -1 if self.y > moveto_y else 1
			for i in range(self.y + step, moveto_y, step):
				if arr[self.x][i] != 0:
					return False
			return True
		# 判断向y轴方向移动
		if self.y == moveto_y:
			step = -1 if self.x > moveto_x else 1
			for i in range(self.x + step, moveto_x, step):
				if arr[i][self.y] != 0:
					return False
			return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 11
		return score


class Horse(Pieces):
	"""
	棋: 马
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player, x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		move_x = moveto_x-self.x
		move_y = moveto_y - self.y
		# 判断马移动方向是否在合法范围内
		if abs(move_x) == 1 and abs(move_y) == 2:
			step = 1 if move_y > 0 else -1
			if arr[self.x][self.y + step] == 0:
				return True
		if abs(move_x) == 2 and abs(move_y) == 1:
			step = 1 if move_x > 0 else -1
			if arr[self.x + step][self.y] == 0:
				return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 5
		return score


class Elephants(Pieces):
	"""
	棋: 相/象
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player, x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		if self.y <= 4 and moveto_y >= 5 or self.y >= 5 and moveto_y <= 4:
			return False
		move_x = moveto_x - self.x
		move_y = moveto_y - self.y
		# 判断相/象合法移动范围
		if abs(move_x) == 2 and abs(move_y) == 2:
			step_x = 1 if move_x > 0 else -1
			step_y = 1 if move_y > 0 else -1
			if arr[self.x + step_x][self.y + step_y] == 0:
				return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 2
		return score


class Scholar(Pieces):
	"""
	棋: 仕/士
	"""
	def __init__(self, player,  x, y):
		self.player = player
		super().__init__(player,  x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		if moveto_x < 3 or moveto_x > 5:
			return False
		if 2 < moveto_y < 7:
			return False
		# 判断仕/士合法移动范围
		move_x = moveto_x - self.x
		move_y = moveto_y - self.y
		if abs(move_x) == 1 and abs(move_y) == 1:
			return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 2
		return score


class King(Pieces):
	"""
	棋: 将/帅
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player, x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		if moveto_x < 3 or moveto_x > 5:
			return False
		if 2 < moveto_y < 7:
			return False
		# 判断将/帅合法移动范围
		move_x = moveto_x - self.x
		move_y = moveto_y - self.y
		if abs(move_x) + abs(move_y) == 1:
			return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 150
		return score


class Cannons(Pieces):
	"""
	棋: 炮
	"""
	def __init__(self, player,  x, y):
		self.player = player
		super().__init__(player, x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		over_flag = False
		if self.x == moveto_x:
			step = -1 if self.y > moveto_y else 1
			for i in range(self.y + step, moveto_y, step):
				if arr[self.x][i] != 0:
					if over_flag:
						return False
					else:
						over_flag = True
			if over_flag and arr[moveto_x][moveto_y] == 0:
				return False
			if not over_flag and arr[self.x][moveto_y] != 0:
				return False
			return True
		if self.y == moveto_y:
			step = -1 if self.x > moveto_x else 1
			for i in range(self.x + step, moveto_x, step):
				if arr[i][self.y] != 0:
					if over_flag:
						return False
					else:
						over_flag = True
			if over_flag and arr[moveto_x][moveto_y] == 0:
				return False
			if not over_flag and arr[moveto_x][self.y] != 0:
				return False
			return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 6
		return score


class Pawns(Pieces):
	"""
	棋: 卒/兵
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player,  x, y)

	def can_move(self, arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		if self.x == moveto_x and self.y == moveto_y:
			return False
		if arr[moveto_x][moveto_y] == self.player:
			return False
		move_x = moveto_x - self.x
		move_y = moveto_y - self.y
		# 判断卒/兵合法移动位置
		if self.player == 1:
			if self.y > 4 and move_x != 0:
				return False
			if move_y > 0:
				return False
		elif self.player == 2:
			if self.y <= 4 and move_x != 0:
				return False
			if move_y < 0:
				return False
		if abs(move_x) + abs(move_y) == 1:
			return True

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		score = 2
		return score