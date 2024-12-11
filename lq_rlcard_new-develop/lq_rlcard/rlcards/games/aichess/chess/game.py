# -*- coding: utf-8 -*-

import gif
import time
import rules
import pieces
import pygame
import alpha_beta

from button import Button
from rlcards.games.aichess.chess.utils import list_pieces_to_arr


class MainGame:
	"""
	游戏
	"""
	window = None
	Start_X = gif.Start_X
	Start_Y = gif.Start_Y
	Line_Span = gif.Line_Span

	Max_X = Start_X + 8 * Line_Span
	Max_Y = Start_Y + 9 * Line_Span

	from_x = 0
	from_y = 0
	to_x = 0
	to_y = 0

	click_x = -1
	click_y = -1

	# 其中包括了棋盘的生成
	mg_init = alpha_beta.AlphaBeta()  # alpha_beta剪枝流程
	player1Color = gif.player1Color  # 玩家2颜色
	player2Color = gif.player2Color  # 玩家1颜色
	Put_down_flag = player1Color
	piecesSelected = None

	button_go = None
	piecesList = []

	def start_game(self):
		"""
		开始游戏
		"""
		# 设置界面大小
		MainGame.window = pygame.display.set_mode([gif.SCREEN_WIDTH, gif.SCREEN_HEIGHT])
		pygame.display.set_caption("中国象棋")
		# 创建button按钮
		MainGame.button_go = Button(MainGame.window, "重新开始", gif.SCREEN_WIDTH - 200, 300)
		self.piece_init()

		# 循环获取事件[整个流程]
		while True:
			time.sleep(0.1)
			MainGame.window.fill(gif.BG_COLOR)
			self.draw_chess_board()
			MainGame.button_go.draw_button()
			self.pieces_display()
			self.victory_or_defeat()
			self.calc_move_rules()
			self.get_event()
			pygame.display.update()
			pygame.display.flip()

	@staticmethod
	def draw_chess_board():
		"""
		绘制棋盘线条
		"""
		mid_end_y = MainGame.Start_Y + 4 * MainGame.Line_Span  # 50 + 4 * 60
		min_start_y = MainGame.Start_Y + 5 * MainGame.Line_Span  # 50 + 5 * 60
		# x方向9条线
		for i in range(0, 9):
			x = MainGame.Start_X + i * MainGame.Line_Span
			if i == 0 or i == 8:
				#  y = MainGame.Start_Y + i * MainGame.Line_Span
				pygame.draw.line(MainGame.window, gif.BLACK, [x, MainGame.Start_Y], [x, MainGame.Max_Y], 1)
			else:
				pygame.draw.line(MainGame.window, gif.BLACK, [x, MainGame.Start_Y], [x, mid_end_y], 1)
				pygame.draw.line(MainGame.window, gif.BLACK, [x, min_start_y], [x, MainGame.Max_Y], 1)
		# y轴方向10条线
		for i in range(0, 10):
			# x = MainGame.Start_X + i * MainGame.Line_Span
			y = MainGame.Start_Y + i * MainGame.Line_Span
			pygame.draw.line(MainGame.window, gif.BLACK, [MainGame.Start_X, y], [MainGame.Max_X, y], 1)
		speed_dial_start_x = MainGame.Start_X + 3 * MainGame.Line_Span
		speed_dial_end_x = MainGame.Start_X + 5 * MainGame.Line_Span
		speed_dial_y1 = MainGame.Start_Y + 0 * MainGame.Line_Span
		speed_dial_y2 = MainGame.Start_Y + 2 * MainGame.Line_Span
		speed_dial_y3 = MainGame.Start_Y + 7 * MainGame.Line_Span
		speed_dial_y4 = MainGame.Start_Y + 9 * MainGame.Line_Span
		pygame.draw.line(
			MainGame.window,
			gif.BLACK,
			[speed_dial_start_x, speed_dial_y1],
			[speed_dial_end_x, speed_dial_y2],
			1
		)

		pygame.draw.line(
			MainGame.window,
			gif.BLACK,
			[speed_dial_start_x, speed_dial_y2],
			[speed_dial_end_x, speed_dial_y1],
			1
		)

		pygame.draw.line(
			MainGame.window,
			gif.BLACK,
			[speed_dial_start_x, speed_dial_y3],
			[speed_dial_end_x, speed_dial_y4],
			1
		)

		pygame.draw.line(
			MainGame.window,
			gif.BLACK,
			[speed_dial_start_x, speed_dial_y4],
			[speed_dial_end_x, speed_dial_y3],
			1
		)

	@staticmethod
	def piece_init():
		"""
		条初始化
		"""
		MainGame.piecesList.append(pieces.Rooks(MainGame.player2Color, 0, 0))
		MainGame.piecesList.append(pieces.Rooks(MainGame.player2Color, 8, 0))
		MainGame.piecesList.append(pieces.Elephants(MainGame.player2Color, 2, 0))
		MainGame.piecesList.append(pieces.Elephants(MainGame.player2Color, 6, 0))
		MainGame.piecesList.append(pieces.King(MainGame.player2Color, 4, 0))
		MainGame.piecesList.append(pieces.Horse(MainGame.player2Color, 1, 0))
		MainGame.piecesList.append(pieces.Horse(MainGame.player2Color, 7, 0))
		MainGame.piecesList.append(pieces.Cannons(MainGame.player2Color, 1, 2))
		MainGame.piecesList.append(pieces.Cannons(MainGame.player2Color, 7, 2))
		MainGame.piecesList.append(pieces.Scholar(MainGame.player2Color, 3, 0))
		MainGame.piecesList.append(pieces.Scholar(MainGame.player2Color, 5, 0))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player2Color, 0, 3))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player2Color, 2, 3))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player2Color, 4, 3))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player2Color, 6, 3))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player2Color, 8, 3))
		MainGame.piecesList.append(pieces.Rooks(MainGame.player1Color, 0, 9))
		MainGame.piecesList.append(pieces.Rooks(MainGame.player1Color, 8, 9))
		MainGame.piecesList.append(pieces.Elephants(MainGame.player1Color, 2, 9))
		MainGame.piecesList.append(pieces.Elephants(MainGame.player1Color, 6, 9))
		MainGame.piecesList.append(pieces.King(MainGame.player1Color, 4, 9))
		MainGame.piecesList.append(pieces.Horse(MainGame.player1Color, 1, 9))
		MainGame.piecesList.append(pieces.Horse(MainGame.player1Color, 7, 9))
		MainGame.piecesList.append(pieces.Cannons(MainGame.player1Color, 1, 7))
		MainGame.piecesList.append(pieces.Cannons(MainGame.player1Color, 7, 7))
		MainGame.piecesList.append(pieces.Scholar(MainGame.player1Color, 3, 9))
		MainGame.piecesList.append(pieces.Scholar(MainGame.player1Color, 5, 9))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player1Color, 0, 6))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player1Color, 2, 6))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player1Color, 4, 6))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player1Color, 6, 6))
		MainGame.piecesList.append(pieces.Pawns(MainGame.player1Color, 8, 6))

	@staticmethod
	def pieces_display():
		"""
		显示条
		"""
		for item in MainGame.piecesList:
			item.display_pieces(MainGame.window)

	def get_event(self):
		"""
		获取点击事件
		"""
		event_list = pygame.event.get()
		for event in event_list:
			if event.type == pygame.QUIT:
				self.end_game()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				pos = pygame.mouse.get_pos()
				mouse_x = pos[0]
				mouse_y = pos[1]
				if (mouse_x > MainGame.Start_X - MainGame.Line_Span / 2 and mouse_x < MainGame.Max_X + MainGame.Line_Span / 2) and \
						(mouse_y > MainGame.Start_Y - MainGame.Line_Span / 2 and mouse_y < MainGame.Max_Y + MainGame.Line_Span / 2):
					if MainGame.Put_down_flag != MainGame.player1Color:
						return
					click_x = round((mouse_x - MainGame.Start_X) / MainGame.Line_Span)
					click_y = round((mouse_y - MainGame.Start_Y) / MainGame.Line_Span)
					click_mod_x = (mouse_x - MainGame.Start_X) % MainGame.Line_Span
					click_mod_y = (mouse_y - MainGame.Start_Y) % MainGame.Line_Span
					if abs(click_mod_x - MainGame.Line_Span / 2) >= 5 and abs(click_mod_y - MainGame.Line_Span / 2) >= 5:
						self.from_x = MainGame.click_x
						self.from_y = MainGame.click_y
						self.to_x = click_x
						self.to_y = click_y
						MainGame.click_x = click_x
						MainGame.click_y = click_y
						self.put_down_pieces(click_x, click_y)
				else:
					print("out click")
				if MainGame.button_go.is_click():
					print("重置对局，开始新游戏对局")
				else:
					print("真人玩家移动棋子: 开始落棋")

	def put_down_pieces(self, x, y):
		"""
		在线条上落棋
		"""
		# 计算出当前作为的棋子[object]
		select_filter = list(filter(
			lambda cm: cm.x == x and cm.y == y and cm.player == MainGame.player1Color, MainGame.piecesList)
		)
		if len(select_filter):
			MainGame.piecesSelected = select_filter[0]
			return
		if MainGame.piecesSelected:
			arr = list_pieces_to_arr(MainGame.piecesList)
			# for i in arr:
			# 	print("输出棋盘局面: ", i)
			if MainGame.piecesSelected.can_move(arr, x, y):
				self.piece_move(MainGame.piecesSelected, x, y)
				MainGame.Put_down_flag = MainGame.player2Color
		else:
			fi = filter(lambda p: p.x == x and p.y == y, MainGame.piecesList)
			list_fi = list(fi)
			if len(list_fi) != 0:
				MainGame.piecesSelected = list_fi[0]

	@staticmethod
	def piece_move(pieces, x, y):
		"""
		更新当前棋盘中棋子移动后的盘局
		"""
		print("移动棋子之前位置[move to]: " + "x: " + str(pieces.x) + " " + "y: " + str(pieces.y))
		for item in MainGame.piecesList:
			# 移除掉已被吃掉的棋子
			if item.x == x and item.y == y:
				MainGame.piecesList.remove(item)
		# 更新棋子坐标[车、马、象/相、仕/士、帅/将、兵/卒]
		pieces.x = x
		pieces.y = y
		print("移动棋子之后位置[move to]: " + "x: " + str(x) + " " + "y: " + str(y))

	def calc_move_rules(self):
		"""
		计算机器人移动规则
		"""
		if MainGame.Put_down_flag == MainGame.player2Color:
			print()
			print("========轮到电脑了========")
			start_time = time.time()
			# 根据真人玩家移动计算机器人最佳移动位置
			# 真人玩家: [原始位置x: 1, 原始位置y: 2, 移动位置x: 1, 移动位置y: 4]
			# 机器人: move_rules[2], move_rules[3]
			move_rules = rules.calc_legal_move(
				MainGame.piecesList,  # 棋盘局面
				self.from_x,  # 玩家原始位置x
				self.from_y,  # 玩家原始位置y
				self.to_x,  # 玩家移动位置x
				self.to_y,  # 玩家移动位置y
				self.mg_init
			)
			if rules is None:
				return
			piece_move = None
			end_time = time.time()
			print("输出计算花费时间: ", end_time - start_time)
			for item in MainGame.piecesList:
				if item.x == move_rules[0] and item.y == move_rules[1]:
					piece_move = item
			self.piece_move(piece_move, move_rules[2], move_rules[3])
			# for items in self.mg_init.board.board:
				# print("输出item[belong, chess_type]: ", [[item.belong, item.chess_type] for item in items])
			MainGame.Put_down_flag = MainGame.player1Color

	def victory_or_defeat(self):
		"""
		判断游戏是否胜利
		"""
		result = [MainGame.player1Color, MainGame.player2Color]
		for item in MainGame.piecesList:
			# 判断将/帅是否被吃掉
			if type(item) == pieces.King:
				if item.player == MainGame.player1Color and MainGame.player1Color in result:
					result.remove(MainGame.player1Color)
				if item.player == MainGame.player2Color and MainGame.player2Color in result:
					result.remove(MainGame.player2Color)
		# 当前对局还没结束
		if len(result) == 0:
			return
		# 判断当前的赢家[真人玩家: 1] -> win/lose
		if result[0] == MainGame.player1Color:
			txt = "失败！"
		else:
			txt = "胜利！"
		MainGame.window.blit(self.get_text_surface("%s" % txt), (gif.SCREEN_WIDTH - 300, 300))
		MainGame.Put_down_flag = gif.overColor

	@staticmethod
	def get_text_surface(text):
		pygame.font.init()
		font = pygame.font.SysFont('kaiti', 40)
		txt = font.render(text, True, gif.TEXT_COLOR)
		return txt

	@staticmethod
	def end_game():
		print("exit")
		exit()

if __name__ == '__main__':
	MainGame().start_game()