# -*- coding: utf-8 -*-

from rlcards.games.aichess.xqengine.const import FILE_LEFT, RANK_TOP
from rlcards.games.aichess.xqengine.util import new_chr, asc, rank_x, rank_y, src, dst

"""
棋子用变量sqSelected表示，sq代表格子的编号。判断棋子uc_pc_Squares[sq]是否被选中，只需要判断sq与sqSelected是否相等即可
sqSelected == 0表示没有棋子被选中

一个走法只用一个数子表示，即mv=sqSrc + sqDst * 256，mv代表走法，mv%256就是起始格子的编号，mv/256就是目标格子的编号
走完一步棋后，通常会把该走法赋值给变量mvLast，并把mvLast%256和mvLast/256这两个格子都做上标记，用旋转的位棋盘计算受车攻击的格子
"""

def move2icc(mv):
	"""
	转换更新象棋坐标表示[原始坐标，目标坐标]
	把字符串型Move转换成运算型Move
	"""
	sq_src = src(mv)  # 获得走法的起点
	sq_dst = dst(mv)  # 获得走法的终点
	return new_chr(
		asc("A") + rank_x(sq_src) - FILE_LEFT  # 获取格子纵坐标
	) + new_chr(
		asc("9") - rank_y(sq_src) + RANK_TOP  # 获取格子的横坐标
	) + "-" + new_chr(
		asc("A") + rank_x(sq_dst) - FILE_LEFT
	) + new_chr(
		asc("9") - rank_y(sq_dst) + RANK_TOP
	)

def cord2uit8(cord):
	"""
	解析象棋坐标
	"""
	alphabet = asc(cord[0]) - asc("A") + FILE_LEFT
	numeric = asc("9") - asc(cord[1]) + RANK_TOP
	return (numeric << 4) + alphabet

def icc2move(icc):
	"""
	还原象棋坐标
	"""
	part1 = cord2uit8(icc[3:])
	part2 = cord2uit8(icc[:2])
	return (part1 << 8) + part2