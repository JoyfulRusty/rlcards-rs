# -*- coding: utf-8 -*-

import gif


class Pieces:
	"""
	绘制界面条
	"""
	def __init__(self, player, x, y):
		"""
		初始化参数
		"""
		self.x = x
		self.y = y
		self.player = player
		self.images_key = self.get_image_key()
		self.images = gif.pieces_images[self.images_key]
		self.rect = self.images.get_rect()
		self.rect.left = gif.Start_X + x * gif.Line_Span - self.images.get_rect().width / 2
		self.rect.top = gif.Start_Y + y * gif.Line_Span - self.images.get_rect().height / 2

	def display_pieces(self, screen):
		"""
		渲染界面条
		"""
		self.rect.left = gif.Start_X + self.x * gif.Line_Span - self.images.get_rect().width / 2
		self.rect.top = gif.Start_Y + self.y * gif.Line_Span - self.images.get_rect().height / 2
		screen.blit(self.images, self.rect)

	@staticmethod
	def can_move(arr, moveto_x, moveto_y):
		"""
		判断能否移动
		"""
		pass

	def get_image_key(self):
		"""
		获取图片key
		"""
		return None

	def get_score_weight(self, list_pieces):
		"""
		获取分数权重
		"""
		return None

class Rooks(Pieces):
	"""
	绘制渲染象棋: 车
	"""
	def __init__(self, player, x, y):
		"""
		初始化参数
		"""
		self.player = player
		super().__init__(player, x, y)

	def get_image_key(self):
		"""
		获取图片key
		"""
		if self.player == gif.player1Color:
			# 红车
			return "r_rook"
		else:
			# 黑车
			return "b_rook"

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
	绘制渲染象棋: 马
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player, x, y)

	def get_image_key(self):
		"""
		获取图片key
		"""
		if self.player == gif.player1Color:
			return "r_horse"
		else:
			return "b_horse"

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
	绘制选择象棋: 相/象
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player, x, y)

	def get_image_key(self):
		"""
		获取图像key
		"""
		if self.player == gif.player1Color:
			# 红相
			return "r_elephant"
		else:
			# 黑象
			return "b_elephant"

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
	绘制渲染象棋: 仕/士
	"""
	def __init__(self, player,  x, y):
		self.player = player
		super().__init__(player,  x, y)

	def get_image_key(self):
		"""
		获取图片key
		"""
		if self.player == gif.player1Color:
			# 红仕
			return "r_scholar"
		else:
			# 黑士
			return "b_scholar"

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
	绘制渲染象棋: 将/帅
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player, x, y)

	def get_image_key(self):
		"""
		获取图片key
		"""
		if self.player == gif.player1Color:
			# 红帅
			return "r_king"
		else:
			# 黑将
			return "b_king"

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
	绘制渲染象棋: 炮
	"""
	def __init__(self, player,  x, y):
		self.player = player
		super().__init__(player, x, y)

	def get_image_key(self):
		"""
		获取图片key
		"""
		if self.player == gif.player1Color:
			# 红炮
			return "r_cannon"
		else:
			# 黑炮
			return "b_cannon"

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
	绘制渲染象棋: 卒/兵
	"""
	def __init__(self, player, x, y):
		self.player = player
		super().__init__(player,  x, y)

	def get_image_key(self):
		"""
		获取图片key
		"""
		if self.player == gif.player1Color:
			# 红兵
			return "r_pawn"
		else:
			# 黑卒
			return "b_pawn"

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
		if self.player == gif.player1Color:
			if self.y > 4 and move_x != 0:
				return False
			if move_y > 0:
				return False
		elif self.player == gif.player2Color:
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