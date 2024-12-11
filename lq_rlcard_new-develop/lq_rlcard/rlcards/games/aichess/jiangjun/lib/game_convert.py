# -*- coding: utf-8 -*-

import xmltodict
import numpy as np

from env.cchess_env import x_axis, y_axis, BaseChessBoard, Pos


def board_arr_2_net_input(board_arr, player, feature_list=None):
	"""
	棋盘arr到网络输入
	"""
	if not feature_list:
		feature_list = {
			"black": ['A', 'B', 'C', 'K', 'N', 'P', 'R'],
			"red": ['a', 'b', 'c', 'k', 'n', 'p', 'r']
		}
	# 国际象棋选择器功能
	picker_x = []
	if player == 'b':
		for one in feature_list['red']:
			picker_x.append(np.asarray(board_arr == one, dtype=np.uint8))
		for one in feature_list['black']:
			picker_x.append(np.asarray(board_arr == one, dtype=np.uint8))

	elif player == 'w':
		for one in feature_list['black']:
			picker_x.append(np.asarray(board_arr == one, dtype=np.uint8))
		for one in feature_list['red']:
			picker_x.append(np.asarray(board_arr == one, dtype=np.uint8))

	picker_x = np.asarray(picker_x)
	if player == 'b':
		return picker_x
	elif player == 'w':
		return picker_x[:, ::-1, :]  # plane层面的上下翻转

def convert_game_value(one_file, feature_list, pg_n_2_value):
	"""
	转变游戏价值
	"""
	try:
		pg_n_file = None
		doc = xmltodict.parse(open(onefile, encoding='utf-8').read())
		fen = doc['ChineseChessRecord']["Head"]["FEN"]
		if pg_n_2_value is not None:
			pg_n_file = doc['ChineseChessRecord']["Head"]["From"]
		moves = [i["@value"] for i in doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
		bb = BaseChessBoard(fen)
		if pg_n_2_value is not None:
			val = pg_n_2_value[pg_n_file]
		else:
			place = one_file.split('.')[-2].split('_')[-1]
			if place == 'w':
				val = 1
			elif place == 'b':
				val = -1
			else:
				val = 0
		red = False
		for i in moves:
			red = not red
			x1, y1, x2, y2 = int(i[0]), int(i[1]), int(i[3]), int(i[4])
			new_board_arr = bb.get_board_arr()
			picker_x = []
			if red:
				for one in feature_list['red']:
					picker_x.append(np.asarray(new_board_arr == one, dtype=np.uint8))
				for one in feature_list['black']:
					picker_x.append(np.asarray(new_board_arr == one, dtype=np.uint8))
			else:
				for one in feature_list['black']:
					picker_x.append(np.asarray(new_board_arr == one, dtype=np.uint8))
				for one in feature_list['red']:
					picker_x.append(np.asarray(new_board_arr == one, dtype=np.uint8))
			picker_x = np.asarray(picker_x)
			picker_y = np.asarray([
				x_axis[x1],
				y_axis[y1],
				x_axis[x2],
				y_axis[y2],
			])
			picker_y_rev = np.asarray([
				x_axis[x1],
				y_axis[9 - y1],
				x_axis[x2],
				y_axis[9 - y2],
			])
			if red:
				yield picker_x, picker_y, val
			else:
				yield picker_x[:, ::-1, :], picker_y_rev, -val
			move_result = bb.move(Pos(x1, y1), Pos(x2, y2))
			assert (move_result is not None)
	except ValueError:
		return None