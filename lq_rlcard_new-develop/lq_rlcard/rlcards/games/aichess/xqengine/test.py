# -*- coding: utf-8 -*-

import time
import numpy as np

from rlcards.games.aichess.xqengine.search import Search
from rlcards.games.aichess.xqengine.position import Position
from rlcards.games.aichess.xqengine.chess import move2icc, icc2move


# 用于更新变化棋盘
uni_pieces = {
	4+8: '车', 3+8: '马', 2+8: '相', 1+8: '仕', 0+8: '帅', 6+8: '兵', 5+8: '炮',
	4+16: '俥', 3+16: '傌', 2+16: '象', 1+16: '士', 0+16: '将', 6+16: '卒', 5+16: '砲', 0: '．'
}

def print_board(pos):
	"""
	打印，更新棋盘
	"""
	print()
	print("=====输出对局棋盘变化=====")
	# 制作一个9 x 10的棋盘，并打印输出棋盘布局
	for i, row in enumerate(np.asarray(pos.squares).reshape(16, 16)[3:3+10, 3:3+9]):
		# 绘制棋盘下标y
		print(' ', 9 - i, ''.join(uni_pieces.get(p, p) for p in row))
	# 绘制棋盘下标x
	print('    ａｂｃｄｅｆｇｈｉ\n\n')

# 计算搜索时间
# 不同时间效果对比[(100, 2~3d, 0.5~1.5s), (500, 3~4d, 2~3s), (1000, 4~5d, 4~5s), (1500, 5~6d, 6~10S), (2000, 6~7d, >11s)]
search_time_ms = 1000
pos = Position()
pos.from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
'''
[
	[20 19 18 17 16 17 18 19 20]
	[ 0  0  0  0  0  0  0  0  0]
	[ 0 21  0  0  0  0  0 21  0]
	[22  0 22  0 22  0 22  0 22]
	[ 0  0  0  0  0  0  0  0  0]
	[ 0  0  0  0  0  0  0  0  0]
	[14  0 14  0 14  0 14  0 14]
	[ 0 13  0  0  0  0  0 13  0]
	[ 0  0  0  0  0  0  0  0  0]
	[12 11 10  9  8  9 10 11 12]
]
'''

search = Search(pos, 16)  # 构建搜索类

choice = input("你想要:  \n\t1. 执红先行\n\t2. 执黑后行\n\t 请选择:\n")
assert(choice in ["1", "2"])
mov = None
if choice == "2":
	start_time = time.time()
	mov = search.search_main(64, search_time_ms) # 搜索3秒钟  # 搜索深度和时间
	end_time = time.time()
	print("##输出单个动作搜索耗费时间##1: ", end_time - start_time)
	pos.make_move(mov)
	print("输出to fen1: ", pos.to_fen())

while True:

	print_board(pos)

	# 人来下棋
	if mov:
		print("电脑的上一步: ", move2icc(mov).replace("-", "").lower())
	start_time = time.time()
	hint_mov = search.search_main(64, search_time_ms) # 搜索10毫秒，给出例子
	end_time = time.time()
	print("##输出单个动作搜索耗费时间##2: ", end_time - start_time)
	while True:
		user_step = move2icc(hint_mov)
		if len(user_step) == 4:
			user_step = user_step[:2] + "-" + user_step[2:]
			print("user_step: ", user_step)
		try:
			user_move = icc2move(user_step)
			print("user_move: ", user_move)
			# 判断当前的移动是否合法
			assert(pos.legal_move(user_move))
		except:
			print("你的行棋不合法，请重新输入")
			continue
		pos.make_move(user_move)
		print("输出to fen2: ", pos.to_fen())
		print_board(pos)
		break

	winner = pos.winner()
	if winner is not None:
		if winner == 0:
			print("红方胜利！行棋结束")
		elif winner == 1:
			print("黑方胜利！行棋结束")
		elif winner == 2:
			print("和棋！行棋结束")
		break

	if user_step != "shameonme".upper():
		# 电脑下棋
		start_time = time.time()
		mov = search.search_main(64, search_time_ms) # 搜索3秒钟
		end_time = time.time()
		print("##输出单个动作搜索耗费时间##3: ", end_time - start_time)
		pos.make_move(mov)
		print_board(pos)
		print("输出to fen3: ", pos.to_fen())

	winner = pos.winner()
	if winner is not None:
		if winner == 0:
			print("红方胜利！行棋结束")
		elif winner == 1:
			print("黑方胜利！行棋结束")
		elif winner == 2:
			print("和棋！行棋结束")
		break