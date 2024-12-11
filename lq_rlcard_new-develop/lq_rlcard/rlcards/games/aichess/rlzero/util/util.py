# -*- coding: utf-8 -*-

import copy
import numpy as np

from collections import deque  # 使用队列来判断长将或长捉
from rlcards.games.aichess.rlzero.const import const
from rlcards.games.aichess.rlzero.config import CONFIG

def array2string(array):
	"""
	数组转换为字符
	"""
	return list(filter(lambda string: (const.STRING2ARRAY[string] == array).all(), const.STRING2ARRAY))[0]

def change_state(state_list, move):
	"""
	改变棋盘状态
	move: 字符串'0010'
	"""
	copy_list = copy.deepcopy(state_list)
	y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
	copy_list[toy][tox] = copy_list[y][x]
	copy_list[y][x] = '一一'
	return copy_list

def print_board(_state_array):
	"""
	打印棋盘，可视化用到
	_state_array: [10, 9, 7], HWC
	"""
	board_line = []
	for i in range(10):
		for j in range(9):
			board_line.append(array2string(_state_array[i][j]))
		print(board_line)
		board_line.clear()

def state_list2state_array(state_list):
	"""
	列表棋盘状态到数组棋盘状态
	"""
	_state_array = np.zeros([10, 9, 7])
	for i in range(10):
		for j in range(9):
			_state_array[i][j] = const.STRING2ARRAY[state_list[i][j]]
	return _state_array

def get_all_legal_moves():
	"""
	拿到所有合法走子的集合，长度为2086，也就是神经网络预测的走子概率向量长度
	1.第一个字典: move_id到move_action
	2.第二个字典: move_action到move_id
	3.例: move_id: 0 -> move_action: '0010'
	"""
	_move_id2move_action = {}
	_move_action2move_id = {}
	# 行列，建立交叉点
	row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
	column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	# 士的全部走法
	advisor_labels = [
		'0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
		'9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
	# 象的全部走法
	bishop_labels = [
		'2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
		'2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
		'7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
		'7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
	idx = 0
	for l1 in range(10):
		for n1 in range(9):
			# 马走日
			destinations = \
				[(t, n1) for t in range(10)] + \
				[(l1, t) for t in range(9)] + \
				[(l1 + a, n1 + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]

			for (l2, n2) in destinations:
				if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
					action = column[l1] + row[n1] + column[l2] + row[n2]
					_move_id2move_action[idx] = action
					_move_action2move_id[action] = idx
					idx += 1

	# # 仕/士
	for action in advisor_labels:
		_move_id2move_action[idx] = action
		_move_action2move_id[action] = idx
		idx += 1

	# #  象/相
	for action in bishop_labels:
		_move_id2move_action[idx] = action
		_move_action2move_id[action] = idx
		idx += 1

	return _move_id2move_action, _move_action2move_id

# move_id2move_action, move_action2move_id = get_all_legal_moves()
with open(CONFIG['move_id_dict'], 'r') as f:
	move_action2move_id = eval(f.read())
with open(CONFIG['id_move_dict'], 'r') as f:
	move_id2move_action = eval(f.read())


def flip_map(string):
	"""
	走子反转，用来扩充数据
	"""
	new_str = ''
	for index in range(4):
		if index == 0 or index == 2:
			new_str += (str(string[index]))
		else:
			new_str += (str(8 - int(string[index])))
	return new_str

def check_bounds(to_y, to_x):
	"""
	边界检查
	"""
	if to_y in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
		if to_x in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
			return True
	return False

def check_obstruct(piece, current_player_color):
	"""
	不能走到自己棋子的位置
	当走到的位置存在棋子时，则进行一次判断
	"""
	if piece != '一一':
		if current_player_color == '红':
			# 红棋吃黑棋
			if '黑' in piece:
				return True
			return False
		elif current_player_color == '黑':
			# 黑棋吃红棋
			if '红' in piece:
				return True
			return False
	else:
		return True

def che_sub(state, y, x, raw, col, color, moves):
	"""
	车子部
	"""
	v = state[raw][col]
	if v != '一一':
		cur_color = v[0]
		if cur_color != color:
			moves.append(str(y) + str(x) + str(raw) + str(col))
			return moves, False
		return moves, False
	else:
		moves.append(str(y) + str(x) + str(raw) + str(col))
		return moves, True

def get_che_position(y, x, state, color):
	"""
	todo: 车
	y行10 x列9
	"""
	moves = []
	for i in range(y - 1, -1, -1):
		moves, label = che_sub(state, y, x, i, x, color, moves)
		if not label:
			break
	for i in range(y + 1, 10):
		moves, label = che_sub(state, y, x, i, x, color, moves)
		if not label:
			break
	for i in range(x - 1, -1, -1):
		moves, label = che_sub(state, y, x, y, i, color, moves)
		if not label:
			break
	for i in range(x + 1, 9):
		moves, label = che_sub(state, y, x, y, i, color, moves)
		if not label:
			break
	return moves

def che_one(y, x, state, color):
	"""
	一个车
	"""
	moves = []
	for i in range(y - 1, -1, -1):
		moves, label = che_sub(state, y, x, i, x, color, moves)
		if not label:
			break
	for i in range(y + 1, 10):
		moves, label = che_sub(state, y, x, i, x, color, moves)
		if not label:
			break
	return moves

def get_ma_position(y, x, state, color):
	"""
	todo: 马
	"""
	def get_raw_plus(state, px, py):
		"""
		获取row+
		"""
		piece_close = state[px + 1][py]
		if piece_close != '一一':
			return False
		return True

	def get_raw_minus(state, px, py):
		"""
		获取row-
		"""
		piece_close = state[px - 1][py]
		if piece_close != '一一':
			return False
		return True

	def get_col_plus(state, px, py):
		"""
		获取col+
		"""
		piece_close = state[px][py + 1]
		if piece_close != '一一':
			return False
		return True

	def get_col_minus(state, px, py):
		"""
		获取col-
		"""
		piece_close = state[px][py - 1]
		if piece_close != '一一':
			return False
		return True

	def get_default(state, px, py):
		return True

	temp_dict = {
		2: get_raw_plus,
		-2: get_raw_minus,
		6: get_col_plus,
		-6: get_col_minus
	}

	px, py = y, x
	places = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
	moves = []
	for place in places:
		x = px + place[0]
		y = py + place[1]
		if 0 <= x <= 9 and 0 <= y <= 8:
			f1 = temp_dict.get(place[0], get_default)(state, px, py)
			f2 = temp_dict.get(place[1] * 3, get_default)(state, px, py)
			if f1 and f2:
				piece_other = state[x][y]
				if color != piece_other[0]:
					moves.append(str(px) + str(py) + str(x) + str(y))
	return moves

def get_bishop_position(y, x, state, color):
	"""
	相/象位置，一个17个固定位置，直接使用字典
	"""
	moves = []
	move_list = const.BISHOP_DICT.get((y, x), [])
	for i in move_list:
		move, end_pos, mid_pos = i
		if state[mid_pos[0], mid_pos[1]] == '一一' and state[end_pos[0], end_pos[1]][0] != color:
			moves.append(move)
	return moves

def get_scholar_position(y, x, state, color):
	moves = []
	move_list = const.SCHOLAR_DICT.get((y, x), [])
	for i in move_list:
		move, end_pos = i
		if state[end_pos[0], end_pos[1]][0] != color:
			moves.append(move)
	return moves

def get_king_position(y, x, state, color):
	"""

	"""
	moves = []
	move_list = const.KING_DICT.get((y, x), [])
	opponent = const.OPPOSITE_DICT[color] + '帅'
	for i in move_list:
		move, end_pos = i
		py, px = end_pos
		if state[py][px][0] != color:
			# 搜索王的位置，判断是否王对王，中间存在一个棋子就跳出
			other = 'none'
			if color == '红':
				for j in range(py + 1, 10):
					guy = state[j][px]
					if guy != '一一':
						other = guy
						break
				if other != opponent:
					moves.append(move)
			else:
				for j in range(py - 1, -1, -1):
					guy = state[j][px]
					if guy != '一一':
						other = guy
						break
				if other != opponent:
					moves.append(move)
	return moves

def get_pawn_position(y, x, state, color):
	moves = []
	if color == '红':
		if y <= 4:
			raw = y + 1
			col = x
			if raw <= 9 and state[raw][col][0] != color:
				moves.append(str(y) + str(x) + str(raw) + str(col))
		else:
			for i in [[0, -1], [0, 1], [1, 0]]:
				raw = y + i[0]
				col = x + i[1]
				if raw <= 9 and -1 < col < 9 and state[raw][col][0] != color:
					moves.append(str(y) + str(x) + str(raw) + str(col))
	elif color == '黑':
		if y > 4:
			raw = y - 1
			col = x
			if raw >= 0 and state[raw][col][0] != color:
				moves.append(str(y) + str(x) + str(raw) + str(col))
		else:
			for i in [[0, -1], [0, 1], [-1, 0]]:
				raw = y + i[0]
				col = x + i[1]
				if raw > -1 and -1 < col < 9 and state[raw][col][0] != color:
					moves.append(str(y) + str(x) + str(raw) + str(col))
	return moves

def get_pawn_one_position(y, x, state, color):
	moves = []
	if color == '红':
		if y <= 4:
			raw = y + 1
			col = x
			if raw <= 9 and state[raw][col][0] != color:
				moves.append(str(y) + str(x) + str(raw) + str(col))
		else:
			for i in [[1, 0]]:
				raw = y + i[0]
				col = x + i[1]
				if raw <= 9 and state[raw][col][0] != color:
					moves.append(str(y) + str(x) + str(raw) + str(col))
	elif color == '黑':
		if y > 4:
			raw = y - 1
			col = x
			if raw >= 0 and state[raw][col][0] != color:
				moves.append(str(y) + str(x) + str(raw) + str(col))
		else:
			for i in [[-1, 0]]:
				raw = y + i[0]
				col = x + i[1]
				if raw >= 0 and state[raw][col][0] != color:
					moves.append(str(y) + str(x) + str(raw) + str(col))
	return moves

def get_cannon_position(y, x, state, color):
	opposite = const.OPPOSITE_DICT[color]
	moves = []
	for i in range(y - 1, -1, -1):
		if state[i][x] == '一一':
			moves.append(str(y) + str(x) + str(i) + str(x))
		else:
			for j in range(i - 1, -1, -1):
				v = state[j][x]
				if v[0] == opposite:
					moves.append(str(y) + str(x) + str(j) + str(x))
					break
				elif v[0] != '一':
					break
			break
	for i in range(y + 1, 10):
		if state[i][x] == '一一':
			moves.append(str(y) + str(x) + str(i) + str(x))
		else:
			for j in range(i + 1, 10):
				v = state[j][x]
				if v[0] == opposite:
					moves.append(str(y) + str(x) + str(j) + str(x))
					break
				elif v[0] != '一':
					break
			break
	for i in range(x - 1, -1, -1):
		if state[y][i] == '一一':
			moves.append(str(y) + str(x) + str(y) + str(i))
		else:
			for j in range(i - 1, -1, -1):
				v = state[y][j]
				if v[0] == opposite:
					moves.append(str(y) + str(x) + str(y) + str(j))
					break
				elif v[0] != '一':
					break
			break
	for i in range(x + 1, 9):
		if state[y][i] == '一一':
			moves.append(str(y) + str(x) + str(y) + str(i))
		else:
			for j in range(i + 1, 9):
				v = state[y][j]
				if v[0] == opposite:
					moves.append(str(y) + str(x) + str(y) + str(j))
					break
				elif v[0] != '一':
					break
			break
	return moves

def get_cannon_one_position(y, x, state, color):
	opposite = const.OPPOSITE_DICT[color]
	moves = []
	for i in range(y - 1, -1, -1):
		if state[i][x] == '一一':
			moves.append(str(y) + str(x) + str(i) + str(x))
		else:
			for j in range(i - 1, -1, -1):
				v = state[j][x]
				if v[0] == opposite:
					moves.append(str(y) + str(x) + str(j) + str(x))
					break
				elif v[0] != '一':
					break
			break
	for i in range(y + 1, 10):
		if state[i][x] == '一一':
			moves.append(str(y) + str(x) + str(i) + str(x))
		else:
			for j in range(i + 1, 10):
				v = state[j][x]
				if v[0] == opposite:
					moves.append(str(y) + str(x) + str(j) + str(x))
					break
				elif v[0] != '一':
					break
			break
	return moves

def where_kings(state, color):
	"""
	赋值默认值，避免判断错误
	"""
	top_pos = [0, 1]
	below_pos = [0, 2]

	pos_list = [4, 3, 5]
	for i in [0, 1, 2]:
		for j in pos_list:
			if state[i][j][1] == '帅':
				top_pos = [i, j]
				break
	for i in [9, 8, 7]:
		for j in pos_list:
			if state[i][j][1] == '帅':
				below_pos = [i, j]
				break
	king_move = []
	face_to_face = False
	if top_pos[1] == below_pos[1]:
		face_to_face = True
		col = below_pos[1]
		for i in range(top_pos[0] + 1, below_pos[0]):
			guy = state[i][col]
			if guy != '一一':
				face_to_face = False
				break
	if face_to_face:
		if color == '红':
			king_move.append(str(top_pos[0]) + str(top_pos[1]) + str(below_pos[0]) + str(below_pos[1]))
		else:
			king_move.append(str(below_pos[0]) + str(below_pos[1]) + str(top_pos[0]) + str(top_pos[1]))
	return king_move, top_pos, below_pos

def empty(y, x, state, color):
	"""
	判断空
	"""
	return []

def get_legal_moves(state_deque, current_player_color):
	"""
	获取合法移动
	"""
	color = current_player_color
	state = state_deque[-1]
	moves = []
	state = copy.deepcopy(state)
	king_move, top_pos, below_pos = where_kings(state, color)
	if king_move:
		moves = [move_action2move_id.get(king_move[0], '')]
		return moves
	state = np.array(state)
	for y in range(10):
		for x in range(9):
			if state[y][x][0] == color:
				move_per = func_dict.get(state[y][x][1:], empty)(y, x, state, color)
				for i in move_per:
					if state[int(i[2])][int(i[3])][1] == '帅':
						return [move_action2move_id.get(i, '')]
					else:
						moves.append(move_action2move_id.get(i, ''))
	return moves

func_dict = {
	'车': get_che_position,
	'马': get_ma_position,
	'象': get_bishop_position,
	'士': get_scholar_position,
	'帅': get_king_position,
	'兵': get_pawn_position,
	'炮': get_cannon_position,
}