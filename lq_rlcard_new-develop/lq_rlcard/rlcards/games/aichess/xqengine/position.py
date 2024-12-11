# -*- coding: utf-8 -*-

import math
import random

from rlcards.games.aichess.xqengine.util import *
from rlcards.games.aichess.xqengine.const import *
from rlcards.games.aichess.xqengine.book import BOOK_DATA

from rlcards.games.aichess.xqengine.rc import \
	pre_gen_zob_key_table, \
	pre_gen_zob_lock_table, \
	pre_gen_zob_key_player, \
	pre_gen_zob_lock_player

"""
1.uc: 表示每个元素占一个字节

2.pc: 表示每个棋子标识

3.0: 表示空格没有棋子

4.8~14: 依次表示红方的帅、仕、相、马、车、炮、兵

5.16~22: 依次表示黑方的将、士、象、傌、俥、炮、卒

6.(pc & 8 != 0): 表示红方棋子

7.(pc & 16 != 0): 表示黑方棋子

8.选中棋子用变量sq_selected表示，sq代表格子编号，判断棋子是否被选中uc_pc_squares[sq]，只需判断sq与sq_selected是否相等，sq_selected == 0
表示没有棋子被选中

9.一个走法只用一个数字表示，即mv = sq_src + sq_dst * 256，mv代表走法，mv % 256为起始格子的编号，mv / 256为目标格子的编号，走完一步棋子后，
通常把走法赋值给mv_result，并把mv_last % 256和mv_last / 256这两个格子进行标记
"""


class Position:
	"""
	棋盘位置
	"""
	def __init__(self):
		"""
		初始化参数
		"""
		self.vl_red = 0
		self.vl_black = 0
		self.zob_key = 0
		self.zob_lock = 0
		self.distance = 0
		self.sd_player = 0
		self.squares = []
		self.chk_list = []
		self.mv_list = [0]
		self.pc_list = [0]
		self.key_list = [0]

	def clear_board(self):
		"""
		todo: 清空棋盘
		"""
		self.vl_red = 0
		self.vl_black = 0
		self.zob_key = 0
		self.zob_lock = 0
		self.sd_player = 0
		self.squares = []
		self.squares.extend([0] * 256)

	def update_position(self):
		"""
		更新重置变量
		"""
		self.mv_list = [0]
		self.pc_list = [0]
		self.key_list = [0]
		self.chk_list = [self.checked()]
		self.distance = 0

	def add_piece(self, sq, pc, b_del=None):
		"""
		todo: 添加棋子
			^: 对应位相异时，即一个0一个1时取1，相同时取0，所以结果为00000001，转成十进制也就是1
				101 (5)
			XOR 110 (6)
			-------------------
				011 (3)
		"""
		# 改变棋盘变化[落子]，删除或吃棋则为0
		self.squares[sq] = 0 if b_del else pc
		# todo: 白/黑，计算子力价值
		# 小于16减去8
		if pc < 16:
			pc_adjust = pc - 8
			# 计算红棋子力价值
			self.vl_red += (-PIECE_VALUE[pc_adjust][sq] if b_del else PIECE_VALUE[pc_adjust][sq])
		# 大于16减去16
		else:
			pc_adjust = pc - 16
			# 计算黑棋子力价值
			self.vl_black += (-PIECE_VALUE[pc_adjust][squares_flip(sq)] if b_del else PIECE_VALUE[pc_adjust][squares_flip(sq)])
			pc_adjust += 7
		self.zob_key ^= pre_gen_zob_key_table[pc_adjust][sq]
		self.zob_lock ^= pre_gen_zob_lock_table[pc_adjust][sq]

	def move_piece(self, mv):
		"""
		移动着法
		"""
		sq_src = src(mv)  # 棋子原始索引
		sq_dst = dst(mv)  # 棋子移动索引
		# 更新棋盘标识[256]
		pc = self.squares[sq_dst]  # 获取棋子标识
		self.pc_list.append(pc)
		if pc > 0:
			# 将棋子标识位置的值置为0
			self.add_piece(sq_dst, pc, True)
		# 更新前后棋子标识移动位置
		pc = self.squares[sq_src]
		self.add_piece(sq_src, pc, True)
		self.add_piece(sq_dst, pc, False)
		# 添加棋子移动动作
		self.mv_list.append(mv)

	def undo_move_piece(self):
		"""
		撤销着法
		"""
		# 取出当前棋子移动动作
		mv = self.mv_list.pop()
		# 获取索引和移动位置
		sq_src = src(mv)
		sq_dst = dst(mv)

		# 更新前后棋子标识移动位置
		pc = self.squares[sq_dst]
		self.add_piece(sq_dst, pc, True)
		self.add_piece(sq_src, pc, False)
		pc = self.pc_list.pop()
		if pc > 0:
			# 棋子标识大于0则还原其标识
			self.add_piece(sq_dst, pc, False)

	def change_side(self):
		"""
		更新玩家操作
		"""
		self.sd_player = 1 - self.sd_player
		self.zob_key ^= pre_gen_zob_key_player
		self.zob_lock ^= pre_gen_zob_lock_player

	def make_move(self, mv):
		"""
		构建移动
		"""
		zob_key = self.zob_key
		# 移动棋子
		self.move_piece(mv)
		# 检查当前棋子的移动合法性
		if self.checked():
			self.undo_move_piece()
			return False
		# 添加键值
		self.key_list.append(zob_key)
		self.change_side()
		self.chk_list.append(self.checked())
		self.distance += 1
		return True

	def undo_make_move(self):
		"""
		撤销构建移动
		"""
		self.distance -= 1
		self.chk_list.pop()
		self.change_side()
		self.key_list.pop()
		self.undo_move_piece()

	def null_move(self):
		"""
		todo: 空步裁剪，某些条件不使用
			1.被将军情况
			2.进入残局是，自己一方的子力总价值小于某个阈值
			3.不要连续做两次空步裁剪，否则会导致搜索的退化

		空着就是自己不走而让对手连走两次，即在适当时机调整搜索层数，但是它通过相反的方式来表现，这个思想不是在复杂的局面上延申，而是在简单的局面上减
		少搜索，假设希望搜索一个高出边界的节点(alpha-beta搜索的返回值至少是beta)，空着搜索就是先搜索弃权着法[null move]，即使它通常不是最好的，
		如果弃权着法高出边界，那么真正最好的着法也可能会高出边界，就可以直接返回beta而不是继续再去搜索，要把搜索做得更快，弃权着法搜索的深度通常比常
		规着法浅
		"""
		self.mv_list.append(0)
		self.pc_list.append(0)
		self.key_list.append(self.zob_key)
		self.change_side()
		self.chk_list.append(False)
		self.distance += 1

	def undo_null_move(self):
		"""
		撤销空步裁剪
		"""
		self.distance -= 1
		self.chk_list.pop()
		self.change_side()
		self.key_list.pop()
		self.pc_list.pop()
		self.mv_list.pop()

	def from_fen(self, fen):
		"""
		todo: 在网络通讯中，常常用一种FEN串的6段式代码来记录局面:
		1.棋盘
		2.走子方
		3.王车易位权
		4.吃过路兵的目标格
		5.可逆着法数以及总回合数，基本上涵盖了国际象棋某个局面的所有信息
		但是FEN字符串无法记录重复局面，因此UCI协议中规定，局面由棋局的最后一个不可逆局面(发生吃子、进兵或升变的局面)和它的后续着法共同表示，
		这样就涵盖了重复局面的信息
		-----------------------------
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
		"""
		# 清空棋盘
		self.clear_board()
		x = FILE_LEFT  # 3
		y = RANK_TOP  # 3
		index = 0
		if index == len(fen):
			self.update_position()
			return
		c = fen[index]
		while c != " ":
			if c == "/":
				x = FILE_LEFT
				y += 1
				if y > RANK_BOTTOM:
					break
			elif "1" <= c <= "9":
				x += (asc(c) - asc("0"))
			elif "A" <= c <= "Z":
				if x <= FILE_RIGHT:
					pt = char_to_piece(c)
					if pt >= 0:
						self.add_piece(coord_xy(x, y), pt + 8)
					x += 1
			elif "a" <= c <= "z":
				if x <= FILE_RIGHT:
					pt = char_to_piece(new_chr(asc(c) + asc("A") - asc("a")))
					if pt >= 0:
						self.add_piece(coord_xy(x, y), pt + 16)
					x += 1
			index += 1
			if index == len(fen):
				self.update_position()
				return
			c = fen[index]
		index += 1
		if index == len(fen):
			self.update_position()
			return
		if self.sd_player == (0 if fen[index] == "b" else 1):
			self.change_side()
		self.update_position()

	def to_fen(self):
		"""
		打印当前棋盘[str]
		"""
		fen = ""
		for y in range(RANK_TOP, RANK_BOTTOM + 1):  # [3, 13]
			k = 0
			for x in range(FILE_LEFT, FILE_RIGHT + 1):  # [3, 12]
				# 获取棋子标识
				pc = self.squares[coord_xy(x, y)]
				if pc > 0:
					if k > 0:
						# 添加棋子位置字符表示
						fen += new_chr(asc("0") + k)
						k = 0
					fen += FEN_PIECE[pc]
				else:
					k += 1
			if k > 0:
				fen += new_chr(asc("0") + k)
			fen += "/"
		return fen[0: len(fen) - 1] + (" r" if self.sd_player == 0 else " b")

	def generate_moves(self, vls=None):
		"""
		生成合法移动动作
		todo: 运算
			^ 两个位相同为0，相异为1
			& 两个位都为1时，结果才为1
			| 两个位都为0时，结果才为0
			~ 0变1，1变0
			>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
			<< 各二进位全部左移若干位，高位丢弃，低位补0
		"""
		mvs = []
		pc_side_tag = side_tag(self.sd_player)  # 检查边界x
		pc_opp_side_tag = opp_side_tag(self.sd_player)  # 检查边界y
		for sq_src in range(256):
			pc_src = self.squares[sq_src]
			if (pc_src & pc_side_tag) == 0:
				continue
			switch_case = pc_src - pc_side_tag
			# todo: 生成将/帅着法
			if switch_case == PIECE_KING:
				for idx in range(4):
					sq_dst = sq_src + KING_DELTA[idx]
					# 检查将/帅前进移动的位置
					if not in_fort(sq_dst):
						continue
					pc_dst = self.squares[sq_dst]
					# 计算将/帅合法移动范围
					if vls is None:
						# todo: 判断棋子方
						# (pc & 8 != 0): 表示红方棋子
						# (pc & 16 != 0): 表示黑方棋子
						if (pc_dst & pc_side_tag) == 0:
							mvs.append(move(sq_src, sq_dst))
					elif (pc_dst & pc_opp_side_tag) != 0:
						mvs.append(move(sq_src, sq_dst))
						vls.append(mvv_lva(pc_dst, 5))

			# todo: 生成仕/士着法
			elif switch_case == PIECE_ADVISOR:
				for idx in range(4):
					sq_dst = sq_src + ADVISOR_DELTA[idx]
					# 检查仕/士前进移动的位置
					if not in_fort(sq_dst):
						continue
					pc_dst = self.squares[sq_dst]
					# 计算仕/士合法移动范围
					if vls is None:
						# todo: 判断棋子方
						# (pc & 8 != 0): 表示红方棋子
						# (pc & 16 != 0): 表示黑方棋子
						if (pc_dst & pc_side_tag) == 0:
							mvs.append(move(sq_src, sq_dst))
					elif (pc_dst & pc_opp_side_tag) != 0:
						mvs.append(move(sq_src, sq_dst))
						vls.append(mvv_lva(pc_dst, 1))

			# todo: 生成相/象着法
			elif switch_case == PIECE_BISHOP:
				for idx in range(4):
					sq_dst = sq_src + ADVISOR_DELTA[idx]
					# 检查相/象前进移动的位置[象不能过河]，可能卡象眼的棋子位置
					if not (in_board(sq_dst) and home_half(sq_dst, self.sd_player) and self.squares[sq_dst] == 0):
						continue
					sq_dst += ADVISOR_DELTA[idx]
					pc_dst = self.squares[sq_dst]
					if vls is None:
						# todo: 判断棋子方
						# (pc & 8 != 0): 表示红方棋子
						# (pc & 16 != 0): 表示黑方棋子
						if (pc_dst & pc_side_tag) == 0:
							mvs.append(move(sq_src, sq_dst))
					elif (pc_dst & pc_opp_side_tag) != 0:
						mvs.append(move(sq_src, sq_dst))
						vls.append(mvv_lva(pc_dst, 1))

			# todo: 生成马/傌🐎着法
			elif switch_case == PIECE_KNIGHT:
				for idx in range(4):
					sq_dst = sq_src + KING_DELTA[idx]
					# 可能蹩马腿的棋子位置
					if self.squares[sq_dst] > 0:
						continue
					for idy in range(2):
						sq_dst = sq_src + KNIGHT_DELTA[idx][idy]
						# 检查马/傌🐎前进移动位置[马/傌🐎能过河]
						if not in_board(sq_dst):
							continue
						pc_dst = self.squares[sq_dst]
						if vls is None:
							# todo: 判断棋子方
							# (pc & 8 != 0): 表示红方棋子
							# (pc & 16 != 0): 表示黑方棋子
							if (pc_dst & pc_side_tag) == 0:
								mvs.append(move(sq_src, sq_dst))
						elif (pc_dst & pc_opp_side_tag) != 0:
							mvs.append(move(sq_src, sq_dst))
							vls.append(mvv_lva(pc_dst, 1))

			# todo: 生成车/俥着法
			elif switch_case == PIECE_ROOK:
				for idx in range(4):
					delta = KING_DELTA[idx]
					sq_dst = sq_src + delta
					# 检查车/俥前进移动位置[车/俥能过河]
					while in_board(sq_dst):
						pc_dst = self.squares[sq_dst]
						if pc_dst == 0:
							if vls is None:
								mvs.append(move(sq_src, sq_dst))
						else:
							if (pc_dst & pc_opp_side_tag) != 0:
								mvs.append(move(sq_src, sq_dst))
								if vls is not None:
									vls.append(mvv_lva(pc_dst, 4))
							break
						sq_dst += delta

			# todo: 生成炮着法
			elif switch_case == PIECE_CANNON:
				for idx in range(4):
					delta = KING_DELTA[idx]
					sq_dst = sq_src + delta
					# 检查炮前进移动位置[炮能过河]
					while in_board(sq_dst):
						pc_dst = self.squares[sq_dst]
						# 没有炮架，没有炮架则直接移动
						if pc_dst == 0:
							if vls is None:
								mvs.append(move(sq_src, sq_dst))
						else:
							break
						sq_dst += delta
					# 存在炮架，则判断是否能够吃棋
					sq_dst += delta
					while in_board(sq_dst):
						pc_dst = self.squares[sq_dst]
						if pc_dst > 0:
							if (pc_dst & pc_opp_side_tag) != 0:
								mvs.append(move(sq_src, sq_dst))
								if vls is not None:
									vls.append(mvv_lva(pc_dst, 4))
							break
						sq_dst += delta

			# todo: 生成兵/卒着法
			elif switch_case == PIECE_PAWN:
				sq_dst = square_forward(sq_src, self.sd_player)
				# 判断未过河
				if in_board(sq_dst):
					pc_dst = self.squares[sq_dst]
					if vls is None:
						# 是否能吃棋子
						if (pc_dst & pc_side_tag) == 0:
							mvs.append(move(sq_src, sq_dst))
					# 不能吃自己的棋子
					elif (pc_dst & pc_opp_side_tag) != 0:
						mvs.append(move(sq_src, sq_dst))
						vls.append(mvv_lva(pc_dst, 2))
				# 判断过河卒/兵
				if away_half(sq_src, self.sd_player):
					for delta in range(-1, 2, 2):
						sq_dst = sq_src + delta
						if in_board(sq_dst):
							pc_dst = self.squares[sq_dst]
							if vls is None:
								# 是否能吃棋子
								if (pc_dst & pc_side_tag) == 0:
									mvs.append(move(sq_src, sq_dst))
							# 不能吃自己的棋子
							elif (pc_dst & pc_opp_side_tag) != 0:
								mvs.append(move(sq_src, sq_dst))
								vls.append(mvv_lva(pc_dst, 2))
		return mvs

	def legal_move(self, mv):
		"""
		判断合法移动动作
		todo: 运算
			^ 两个位相同为0，相异为1
			& 两个位都为1时，结果才为1
			| 两个位都为0时，结果才为0
			~ 0变1，1变0
			>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
			<< 各二进位全部左移若干位，高位丢弃，低位补0
		"""
		sq_src = src(mv)  # todo: 获取走法的起点
		pc_src = self.squares[sq_src]  # 取出棋子值
		# 判断当前玩家边界
		pc_side = side_tag(self.sd_player)
		# todo: 判断棋子方
		# (pc & 8 != 0): 表示红方棋子
		# (pc & 16 != 0): 表示黑方棋子
		if (pc_src & pc_side) == 0:
			return False

		sq_dst = dst(mv)  # todo: 获取走法的终点
		pc_dst = self.squares[sq_dst]
		if (pc_dst & pc_side) != 0:
			return False

		switch_case = pc_src - pc_side
		# todo: 将/帅
		if switch_case == PIECE_KING:
			return in_fort(sq_dst) and king_span(sq_src, sq_dst)

		# todo: 仕/士
		elif switch_case == PIECE_ADVISOR:
			return in_fort(sq_dst) and advisor_span(sq_src, sq_dst)

		# todo: 象/相
		elif switch_case == PIECE_BISHOP:
			return same_half(sq_src, sq_dst) and bishop_span(sq_src, sq_dst) and self.squares[bishop_pin(sq_src, sq_dst)] == 0

		# todo: 马/傌🐎
		elif switch_case == PIECE_KNIGHT:
			sq_pin = knight_pin(sq_src, sq_dst)
			return sq_pin != sq_src and self.squares[sq_pin] == 0

		# todo: 车/俥 or 炮
		elif switch_case == PIECE_ROOK or switch_case == PIECE_CANNON:
			if same_rank(sq_src, sq_dst):
				delta = -1 if sq_dst < sq_src else 1
			elif same_file(sq_src, sq_dst):
				delta = -16 if sq_dst < sq_src else 16
			else:
				return False
			sq_pin = sq_src + delta
			while sq_pin != sq_dst and self.squares[sq_pin] == 0:
				sq_pin += delta
			if sq_pin == sq_dst:
				return pc_dst == 0 or pc_src - pc_side == PIECE_ROOK
			if pc_dst == 0 or pc_src - pc_side != PIECE_CANNON:
				return False
			sq_pin += delta
			while sq_pin != sq_dst and self.squares[sq_pin] == 0:
				sq_pin += delta
			return sq_pin == sq_dst

		# todo: 兵/卒
		elif switch_case == PIECE_PAWN:
			if away_half(sq_dst, self.sd_player) and (sq_dst == sq_src - 1 or sq_dst == sq_src + 1):
				return True
			return sq_dst == square_forward(sq_src, self.sd_player)
		else:
			return False

	def checked(self):
		"""
		检查，重复检测[局面]
		"""
		pc_side = side_tag(self.sd_player)
		pc_opp_side = opp_side_tag(self.sd_player)
		for sq_src in range(0, 256):
			if self.squares[sq_src] != pc_side + PIECE_KING:
				continue
			if self.squares[square_forward(sq_src, self.sd_player)] == pc_opp_side + PIECE_PAWN:
				return True
			for delta in range(-1, 2, 2):
				if self.squares[sq_src + delta] == pc_opp_side + PIECE_PAWN:
					return True
			for idx in range(4):
				if self.squares[sq_src + ADVISOR_DELTA[idx]] != 0:
					continue
				for idy in range(2):
					pc_dst = self.squares[sq_src + KNIGHT_CHECK_DELTA[idx][idy]]
					if pc_dst == pc_opp_side + PIECE_KNIGHT:
						return True
			for idx in range(4):
				delta = KING_DELTA[idx]
				sq_dst = sq_src + delta
				while in_board(sq_dst):
					pc_dst = self.squares[sq_dst]
					if pc_dst > 0:
						if pc_dst == pc_opp_side + PIECE_ROOK or pc_dst == pc_opp_side + PIECE_KING:
							return True
						break
					sq_dst += delta
				sq_dst += delta
				while in_board(sq_dst):
					pc_dst = self.squares[sq_dst]
					if pc_dst > 0:
						if pc_dst == pc_opp_side + PIECE_CANNON:
							return True
						break
					sq_dst += delta
			return False
		return False

	def is_mate(self):
		"""
		todo: 判断是否被将死
		"""
		mvs = self.generate_moves(None)
		for idx in range(len(mvs)):
			# 判断是否还能移动棋子
			if self.make_move(mvs[idx]):
				self.undo_make_move()
				return False
		return True

	def mate_value(self):
		"""
		将死值
		"""
		return self.distance - MATE_VALUE

	def ban_value(self):
		"""
		输棋值
		"""
		return self.distance - BAN_VALUE

	def draw_value(self):
		"""
		和棋值
		"""
		return -DRAW_VALUE if (self.distance & 1) == 0 else DRAW_VALUE

	def evaluate(self):
		"""
		评估函数
		"""
		vl = (self.vl_red - self.vl_black if self.sd_player == 0 else self.vl_black - self.vl_red) + ADVANCED_VALUE
		return vl - 1 if vl == self.draw_value() else vl

	def null_okay(self):
		"""
		null值
		"""
		return (self.vl_red if self.sd_player == 0 else self.vl_black) > NULL_OKAY_MARGIN

	def null_safe(self):
		"""
		空安全值
		"""
		return (self.vl_red if self.sd_player == 0 else self.vl_black) > NULL_SAFE_MARGIN

	def in_check(self):
		"""
		检查
		"""
		return self.chk_list[len(self.chk_list) - 1]

	def captured(self):
		"""
		捕获、攻取[吃棋]
		"""
		return self.pc_list[len(self.pc_list) - 1] > 0

	def rep_value(self, vl_rep):
		"""
		估算值
		"""
		vl_return = (0 if (vl_rep & 2) == 0 else self.ban_value()) + (0 if (vl_rep & 4) == 0 else -self.ban_value())
		return self.draw_value() if vl_return == 0 else vl_return

	def rep_status(self, re_cur):
		"""
		todo: 估算状态
			检查重复局面，返回值[return 1 + (b_pre_check ? 2:0) + (b_opp_pre_check ? 4:0)]
			b_pre_check[本方长将]和b_opp_pre_check[对方长将]都设置为True
			当一方存在非将走法时，则改为False，返回值存在几种可能性:
				A.返回0，表示没有重复局面
				B.返回1，表示存在重复局面，但双方都无长将(判和)
				C.返回3(=1+2)，表示存在重复局面，本方单方面长将(判本方负)
				D.返回5(=1+4)，表示存在重复局面，对方单方面长将(判对方负)
				E.返回7(=1+2+4)，表示存在重复局面，双方长将(判和)
		"""
		recur = re_cur
		self_side = False
		prep_check = True
		opp_prep_check = True
		index = len(self.mv_list) - 1
		while self.mv_list[index] > 0 and self.pc_list[index] == 0:
			if self_side:
				prep_check = prep_check and self.chk_list[index]
				if self.key_list[index] == self.zob_key:
					recur -= 1
					if recur == 0:
						# 判断长将结果[1, 3, 5, 7]
						return 1 + (2 if prep_check else 0) + (4 if opp_prep_check else 0)
			else:
				opp_prep_check = opp_prep_check and self.chk_list[index]
			self_side = not self_side
			index -= 1
		return 0

	def mirror(self):
		"""
		镜像
		"""
		pos = Position()
		pos.clear_board()
		for sq in range(256):
			pc = self.squares[sq]
			if pc > 0:
				# 添加对称局面的棋子
				pos.add_piece(mirror_squares(sq), pc)
		if self.sd_player == 1:
			pos.change_side()
		return pos

	def book_move(self):
		"""
		使用开局库选中动作
		"""
		# 开局库无值，则返回0
		if BOOK_DATA is None or len(BOOK_DATA) == 0:
			return 0
		mirror = False
		zob_target = un_singed_right_shift(self.zob_lock, 1)
		index = binary_search(BOOK_DATA, zob_target)
		# 开局库中无法找到动作时，则使用开局库中镜像
		if index < 0:
			mirror = True
			zob_target = un_singed_right_shift(self.mirror().zob_lock, 1)
			index = binary_search(BOOK_DATA, zob_target)
		if index < 0:
			print("####%%开局库不存在合法动作%%####")
			return 0
		index -= 1
		# 如果找到局面，则向前查找第一个着法
		while index >= 0 and BOOK_DATA[index][0] == zob_target:
			index -= 1

		# todo: 开局库镜像中查找合法动作
		# 向后依次读入属于该局面的每个着法
		mvs = []
		vls = []
		value = 0
		index += 1
		while index < len(BOOK_DATA) and BOOK_DATA[index][0] == zob_target:
			# 如果局面是第二趟搜索到的，则着法必须做镜像
			mv = BOOK_DATA[index][1]
			# 原局面和镜像局面各搜索一趟
			mv = mirror_move(mv) if mirror else mv
			if self.legal_move(mv):
				mvs.append(mv)
				vl = BOOK_DATA[index][2]  # MOVE格式转码
				vls.append(vl)
				value += vl
			index += 1
		if value == 0:
			return 0
		value = math.floor(random.random() * value)
		for index in range(len(mvs)):
			value -= vls[index]
			if value < 0:
				break
		return mvs[index]

	def history_index(self, mv):
		"""
		历史索引
		"""
		return ((self.squares[src(mv)] - 8) << 8) + dst(mv)

	def winner(self):
		"""
		计算赢家
		todo: 解析
			长将判负的局面定为BAN_VALUE(MATE_VALUE - 100)，如果某个局面分值在WIN_VALUE(MATE_VALUE - 200)和BAN_VALUE之间，那么这个局面
			就是利用长将判负策略搜索到的局面
		"""
		# todo: 判断是否被将死
		if self.is_mate():
			return 1 - self.sd_player
		pc = side_tag(self.sd_player) + PIECE_KING
		sq_mate = 0
		for sq in range(256):
			if self.squares[sq] == pc:
				sq_mate = sq
				break
		if sq_mate == 0:
			return 1 - self.sd_player
		# 估算状态，返回3(=1+2)，表示存在重复局面，本方单方面长将(判本方负)
		vl_rep = self.rep_status(3)
		if vl_rep > 0:
			# 估算值，是否被长将判断负
			vl_rep = self.rep_value(vl_rep)
			# todo: 双方不变作和
			if -WIN_VALUE < vl_rep < WIN_VALUE:
				return 2
			# todo: 长打作负
			else:
				return self.sd_player
		# todo: 它是子力价值的和，它是双方棋盘上棋子的数量
		has_material = False
		for sq in range(256):
			if in_board(sq) and (self.squares[sq] & 7) > 2:
				has_material = True
				break
		# todo: 无进攻子力做和
		if not has_material:
			return 2
		return