import pygame

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 650
Start_X = 50
Start_Y = 50
Line_Span = 60

player1Color = 1
player2Color = 2
overColor = 3

BG_COLOR = pygame.Color(200, 200, 200)
Line_COLOR = pygame.Color(255, 255, 200)
TEXT_COLOR = pygame.Color(255, 0, 0)

# 定义颜色
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

repeat = 0

pieces_images = {
	# todo: 黑
	'b_rook': pygame.image.load("img/s2/b_c.gif"),  # 车
	'b_elephant': pygame.image.load("img/s2/b_x.gif"),  # 象
	'b_king': pygame.image.load("img/s2/b_j.gif"),  # 将
	'b_horse': pygame.image.load("img/s2/b_m.gif"),  # 马
	'b_scholar': pygame.image.load("img/s2/b_s.gif"),  # 士
	'b_cannon': pygame.image.load("img/s2/b_p.gif"),  # 炮
	'b_pawn': pygame.image.load("img/s2/b_z.gif"),  # 卒

	# todo: 红
	'r_rook': pygame.image.load("img/s2/r_c.gif"),  # 车
	'r_elephant': pygame.image.load("img/s2/r_x.gif"),  # 相
	'r_king': pygame.image.load("img/s2/r_j.gif"),  # 帅
	'r_horse': pygame.image.load("img/s2/r_m.gif"),  # 马
	'r_scholar': pygame.image.load("img/s2/r_s.gif"),  # 仕
	'r_cannon': pygame.image.load("img/s2/r_p.gif"),  # 炮
	'r_pawn': pygame.image.load("img/s2/r_z.gif"),  # 兵
}
