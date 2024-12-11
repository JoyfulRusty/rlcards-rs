# -*- coding: utf-8 -*-

import gif

def list_pieces_to_arr(pieces_list):
	"""
	打印当前棋盘结果
	"""
	arr = [[0 for i in range(10)] for j in range(9)]
	for i in range(0, 9):
		for j in range(0, 10):
			if len(list(filter(lambda cm: cm.x == i and cm.y == j and cm.player == gif.player1Color, pieces_list))):
				arr[i][j] = gif.player1Color
			elif len(list(filter(lambda cm: cm.x == i and cm.y == j and cm.player == gif.player2Color, pieces_list))):
				arr[i][j] = gif.player2Color
	return arr