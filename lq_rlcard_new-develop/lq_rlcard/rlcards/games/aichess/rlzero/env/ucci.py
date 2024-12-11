# -*- coding: utf-8 -*-

import copy
import random
import subprocess

from rlcards.games.aichess.rlzero.const import const
from rlcards.games.aichess.rlzero.config import CONFIG


# todo: 象棋引擎协议类[Chess Engine Protocol -> u u c i]

with open(CONFIG['move_id_dict'], 'r') as f:
	move_action2move_id = eval(f.read())

# 象眼引擎路径
elephant_eye_path_list = CONFIG['elephant_eye_path_list']

def swap_case(a):
	if a.isalpha():
		return a.lower() if a.isupper() else a.upper()
	return a

def swap_cars_by_s2b(a, s2b=False):
	if a.isalpha():
		if s2b:
			a = const.STATE_TO_BOARD_DICT[a]
		else:
			a = const.REPLACE_DICT[a]
		return a.lower() if a.isupper() else a.upper()
	return a

def swap_all(aa):
	return "".join([swap_case(a) for a in aa])

def state_to_fen(state, turns):
	"""
	状态映射为分数
	"""
	state = ''.join([const.STATE_TO_BOARD_DICT[s] if s.isalpha() else s for s in state])
	# 黑
	if turns % 2 == 0:
		return state + f' w - - 0 {turns}'
	# 白
	return state + f' b - - 0 {turns}'

def fen_to_state(fen):
	"""
	分数转换为状态
	"""
	foo = fen.split(' ')
	position = foo[0]

	return "".join([const.REPLACE_DICT[s] if s.isalpha() else s for s in position])

def flip_fen(fen):
	"""
	翻转分数
	"""
	foo = fen.split(' ')
	rows = foo[0].split('/')

	return "/".join(
		[swap_all(reversed(row)) for row in reversed(rows)]
	) + " " + ('w' if foo[1] == 'b' else 'b') + " " + foo[2] + " " + foo[3] + " " + foo[4] + " " + foo[5]

def flip_state(state):
	"""
	翻转状态
	"""
	rows = state.split('/')

	return "/".join([swap_all(reversed(row)) for row in reversed(rows)])

def state_to_board(state):
	"""
	状态映射到棋盘
	"""
	board = [['.' for _ in range(const.BOARD_WIDTH)] for _ in range(const.BOARD_HEIGHT)]
	x = 0
	y = 9
	for k in range(0, len(state)):
		ch = state[k]
		if ch == ' ':
			break
		if ch == '/':
			x = 0
			y -= 1
		elif '1' <= ch <= '9':
			for i in range(int(ch)):
				board[y][x] = '.'
				x = x + 1
		else:
			board[y][x] = swap_cars_by_s2b(ch, s2b=True)
			x = x + 1
	return board

def board_to_state(board):
	"""
	棋盘映射到状态
	"""
	fen = ''
	for i in range(const.BOARD_HEIGHT - 1, -1, -1):
		c = 0
		for j in range(const.BOARD_WIDTH):
			if board[i][j] == '一一':
				c = c + 1
			else:
				if c > 0:
					fen = fen + str(c)
				fen = fen + swap_cars_by_s2b(board[i][j])
				c = 0
		if c > 0:
			fen = fen + str(c)
		if i > 0:
			fen = fen + '/'
	return fen

def board_to_state_old(board):
	"""
	棋盘映射到状态(旧)
	"""
	fen = ''
	for i in range(const.BOARD_HEIGHT - 1, -1, -1):
		c = 0
		for j in range(const.BOARD_WIDTH):
			if board[i][j] == '.':
				c = c + 1
			else:
				if c > 0:
					fen = fen + str(c)
				fen = fen + swap_cars_by_s2b(board[i][j])
				c = 0
		if c > 0:
			fen = fen + str(c)
		if i > 0:
			fen = fen + '/'
	return fen

def parse_cep_move(move):
	"""
	解析象棋引擎move动作
	"""
	x0, x1 = ord(move[0]) - ord('a'), ord(move[2]) - ord('a')
	move = move[1] + str(x0) + move[3] + str(x1)
	return move

def parse_pika_move(move):
	"""
	解析皮卡鱼象棋引擎动作
	"""
	x0, x1 = ord(move[0]) - ord('a'), ord(move[2]) - ord('a')
	move = inmove[1] + str(x0) + move[3] + str(x1)
	return move

def parse_cep_move_old(move):
	"""
	解析象棋引擎move动作(old)
	"""
	x0, x1 = ord(move[0]) - ord('a'), ord(move[2]) - ord('a')
	move = str(x0) + move[1] + str(x1) + move[3]
	return move

def choice_cep(cep: str):
	"""
	todo: 选择象棋引擎
	"""
	if cep == 'elephant':  # 2
		return CONFIG['elephant_eye_path']
	elif cep == 'cyclone':  # 4
		return CONFIG['cyclone_path']
	elif cep == 'pika':  # 1
		return CONFIG['pika_path']
	elif cep == 'bing_he':  # 3
		return CONFIG['bing_he_path']
	return CONFIG['elephant_eye_path']

def get_cep_by_move(fen, time=3):
	"""
	计算象棋引擎做出的move[ElephantEye]
	"""
	# print("====================象眼引擎====================")
	if CONFIG['play_with_cep']:
		elephant_eye_path = choice_cep('elephant')
	else:
		elephant_eye_path = random.choice(elephant_eye_path_list)
	p = subprocess.Popen(
		elephant_eye_path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True
	)
	set_fen = f'position fen {fen}\n'
	set_option1 = f'setoption randomness large\n'
	set_option2 = f'setoption threads 6\n'
	set_option = set_option1 + set_option2
	cmd = 'ucci\n' + set_option + set_fen + f'go time {time * 1000}\n'
	try:
		out, err = p.communicate(cmd, timeout=time + 0.5)
	except subprocess.TimeoutExpired:
		p.kill()
		try:
			out, err = p.communicate()
		except Exception as e:
			print(f"{e}, cmd = {cmd}")
			return get_cep_by_move(fen, time + 1)
	lines = out.split('\n')
	# todo: 当前引擎计算不出最佳结果时
	if lines[-2] == 'nobestmove':  # [elephant.exe, cyclone.exe]
		print("无最优动作!")
		return None, None
	else:
		if elephant_eye_path.split('/')[-1] == 'binghe.exe':
			move = lines[-3].split(' ')[1]
		else:
			move = lines[-2].split(' ')[1]  # 1 -> [elephant.exe, cyclone.exe]
		if move == 'depth':
			move = lines[-1].split(' ')[6]
		move = parse_cep_move(move)
	return move, lines

def get_cep_move_func(board, color=None):
	"""
	获取象棋引擎计算move
	"""
	# 思考时间，默认象棋引擎思考3秒
	think_time = CONFIG['think_time']
	# [棋盘状态， 玩家座位(黑/红)]
	board_map, color_cur = board.curr_map()

	# 判断当前当前玩家棋子花色[红/黑]
	if color:
		color_cur = color
	# [黑1红2]，红棋思考时间加1秒
	if color_cur == '黑':
		turns = 1
	else:
		turns = 2
		think_time += 1

	# 当前棋盘状态映射为字符串
	state = board_to_state(copy.deepcopy(board_map))
	# 转换当前棋盘为对应分数表示
	fen = state_to_fen(state, turns)

	# 计算当前move和lines
	move, lines = get_cep_by_move(fen, time=think_time)

	# 解析并返回当前选择的move和lines
	return move_action2move_id.get(move, None), lines

def get_pika_fish_by_move(fen, time=3):
	"""
	使用皮卡鱼象棋引擎
	"""
	# print("====================皮卡鱼引擎====================")
	# 读取皮卡鱼引擎路径
	cep_path = choice_cep('pika')
	p = subprocess.Popen(
		cep_path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True,
	)
	set_cmd1 = f'setoption name Threads value 6\n'
	set_cmd2 = f'setoption name EvalFile value pikafish.nnue\n'
	set_cmd3 = f'setoption name UCI_ShowWDL value False\n'
	set_cmd4 = f'setoption name MultiPV value 1\n'
	set_cmd5 = f'setoption name Clear Hash\n'
	set_depth = f'go depth 20\n'
	set_cmd = set_cmd1 + set_cmd2 + set_cmd3 + set_cmd4 + set_cmd5
	set_fen = f'position fen {fen}\n'
	cmd = 'uci\n' + set_cmd + set_fen + set_depth + f'go time {time * 1000}\n'
	try:
		out, err = p.communicate(cmd)
	except subprocess.TimeoutExpired:
		p.kill()
		try:
			out, err = p.communicate()
		except Exception as e:
			print(f"{e}, cmd = {cmd}")
			return get_pika_fish_by_move(fen, time + 1)
	lines = out.split('\n')[:-3]

	move = lines[-1].split(' ')[1]
	move = parse_cep_move(move)
	return move, lines

def get_cep_move_func_pika(board, color=None):
	"""
	获取象棋引擎计算move
	"""
	# 思考时间，默认象棋引擎思考3秒
	think_time = CONFIG['think_time']
	# [棋盘状态， 玩家座位(黑/红)]
	board_map, color_cur = board.curr_map()

	# 判断当前当前玩家棋子花色[红/黑]
	if color:
		color_cur = color
	# [黑1红2]，红棋思考时间加1秒
	if color_cur == '黑':
		turns = 1
	else:
		turns = 2
		think_time += 1

	# 当前棋盘状态映射为字符串
	state = board_to_state(copy.deepcopy(board_map))
	# 转换当前棋盘为对应分数表示
	fen = state_to_fen(state, turns)

	# 计算当前move和lines
	move, lines = get_pika_fish_by_move(fen, time=think_time)

	# 解析并返回当前选择的move和lines
	return move_action2move_id.get(move, None), lines