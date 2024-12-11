# -*- coding: utf-8 -*-

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
通常把走法赋值给mv_last，并把mv_last % 256和mv_last / 256这两个格子进行标记

10.一维数组的好就是上下左右关系非常简明——上面一格是sq - 16，下面一格是sq + 16，左面一格是sq - 1，右面一格是sq + 1

11.预置一个常量数组ccInBoard[256]，表示哪些格子在棋盘外(紫色格子，填0)，哪些格子在棋盘内(浅色格子，填1)，所以就没有必要使用x >= X_LEFT && x 
<= X_RIGHT && y >= Y_TOP && y <= Y_BOTTOM之类的语句了，取而代之的是 ccInBoard[sq] != 0

12.遇到将死或困毙的局面时，应该返回nDistance - INFINITY，这样程序就能找到最短的搜索路线。nDistance是当前节点距离根节点的步数，每走一个走法，
nDistance就增加1，每撤消一个走法，nDistance就减少1

13.一个典型的散列函数，先随机产生一张64x13的表，如果棋子y在位置x上，就把表中[x, y]这个数加到散列值上(忽略溢出)[即Zob值]。值得注意的是，当棋子y从
位置x走到位置z时，可以快速地更新散列值：减去[x, y]并加上[z, y]即可

14.左移8位，产生前面的格子(<<8)

15.求被占格子的"非"运算，获取空的格子

16.根据14，15两个位棋盘，按位与运算，获取前面没有被占的格子
"""

from rlcards.games.aichess.xqengine.const import *


def binary_search(src_list: list, target: int):
	"""
	todo: 二分查找
	"""
	low = 0
	high = len(src_list) - 1
	while low <= high:
		mid = (low + high) >> 1  # /2
		if src_list[mid][0] < target:
			low = mid + 1
		elif src_list[mid][0] > target:
			high = mid - 1
		else:
			return mid
	return -1

def shell_sort(mvs, vls):
	"""
	todo: 希尔排序
	"""
	step_level = 1
	while SHELL_STEP[step_level] < len(mvs):
		step_level += 1
	step_level -= 1
	while step_level > 0:
		step = SHELL_STEP[step_level]
		for i in range(len(mvs)):
			mv_best = mvs[i]
			vl_best = vls[i]
			j = i - step
			# 迭代价值
			while j >= 0 and vl_best > vls[j]:
				mvs[j + step] = mvs[j]
				vls[j + step] = vls[j]
				j -= step
			# 更新最好移动值和最好值
			mvs[j + step] = mv_best
			vls[j + step] = vl_best
		step_level -= 1

def in_board(sq):
	"""
	判断棋子是否在棋盘内
	"""
	return IN_BOARD_[sq] != 0

def in_fort(sq):
	"""
	判断将/帅/士/仕合法移动堡垒中
	"""
	return IN_FORT_[sq] != 0

def rank_y(sq):
	"""
	获取格子的纵坐标
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	return sq >> 4

def rank_x(sq):
	"""
	获取格子横坐标
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	return sq & 15

def coord_xy(x, y):
	"""
	根据纵坐标和横坐标获得格子(根据坐标获取各格子)
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	return x + (y << 4)

def squares_flip(sq):
	"""
	翻转格子，棋盘翻转
	"""
	# 所有棋盘位数减去格子编号
	return 254 - sq

def file_flip(x):
	"""
	纵坐标水平镜像
	"""
	return 14 - x

def rank_flip(y):
	"""
	横坐标垂直镜像
	"""
	return 15 - y

def mirror_squares(sq):
	"""
	todo: 对称局面，格子水平镜像
	"""
	return coord_xy(file_flip(rank_x(sq)), rank_y(sq))

def square_forward(sq, sd):
	"""
	更新棋盘格子编号，前进一个格子的位置
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	return sq - 16 + (sd << 5)

def king_span(sq_src, sq_dst):
	"""
	将/帅合法跨度[跨距/范围]，走法是否符合帅(将)的步长
	"""
	return LEGAL_SPAN[sq_dst - sq_src + 256] == 1  # 1 -> king

def advisor_span(sq_src, sq_dst):
	"""
	士/仕合法跨度[跨距/范围]，走法是否符合仕(士)的步长
	"""
	return LEGAL_SPAN[sq_dst - sq_src + 256] == 2  # 2 -> advisor

def bishop_span(sq_src, sq_dst):
	"""
	象/相合法跨度[跨距/范围]，走法是否符合相(象)的步长
	"""
	return LEGAL_SPAN[sq_dst - sq_src + 256] == 3  # 3 -> bishop

def bishop_pin(sq_src, sq_dst):
	"""
	相腿，相(象)眼的位置
	"""
	return (sq_src + sq_dst) >> 1

def knight_pin(sq_src, sq_dst):
	"""
	获取傌/马/将/帅(腿)位置，马腿的位置
	如果返回sq_src，则说明不是马步
	todo: 判断马的某个走法是否符合规则
		sq_pin = king_pin(sq_src, sq_dst)
		return sq_pin != sq_src & uc_pc_squares[sq_pin] == 0
	"""
	return sq_src + KNIGHT_PIN_[sq_dst - sq_src + 256]

def home_half(sq, sd):
	"""
	判断是否为一半棋盘，是否未过河
	todo: 计算棋盘一半
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
		0x80 -> 128

	快速判断某个格子sq是否在棋盘上，当且仅当(sq & 0x08) == 0时
	sq在棋盘上，因为列超出范围时，sq&0x08不为0，行超出范围时,sq&0x08不为0
	"""
	return (sq & 0x80) != (sd << 7)  # 0x80 -> 128

def away_half(sq, sd):
	"""
	棋盘一半，是否已过河
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
		0x80 -> 128

	快速判断某个格子sq是否在棋盘上，当且仅当(sq & 0x08) == 0时
	sq在棋盘上，因为列超出范围时，sq&0x08不为0，行超出范围时,sq&0x08不为0
	"""
	return (sq & 0x80) == (sd << 7)  # 0x80 -> 128

def same_half(sq_src, sq_dst):
	"""
	是否在河的同一边
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
		------------------------------------
		0x80 -> 128
		0001 -> 1
		^
		0010 -> 2
		----
		0011 -> 3

	快速判断某个格子sq是否在棋盘上，当且仅当(sq & 0x08) == 0时
	sq在棋盘上，因为列超出范围时，sq&0x08不为0，行超出范围时,sq&0x08不为0
	"""
	return ((sq_src ^ sq_dst) & 0x80) == 0  # 0x80 -> 128

def same_rank(sq_src, sq_dst):
	"""
	是否在同一行
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
		------------------------------------
		0xf0 -> 240
		0001 -> 1
		^
		0010 -> 2
		----
		0011 -> 3
	"""
	return ((sq_src ^ sq_dst) & 0xf0) == 0  # 0xf0 -> 240

def same_file(sq_src, sq_dst):
	"""
	是否在同一列
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
		------------------------------------
		0x0f -> 15
		0001 -> 1
		^
		0010 -> 2
		----
		0011 -> 3
	"""
	return ((sq_src ^ sq_dst) & 0x0f) == 0  # 0x0f -> 15

def side_tag(sd):
	"""
	判断边界标签
	获取红黑标记，红8，黑16
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
		0000 -> 8421
		00000000 ->128 64 32 16 8421
		0001 -> 1000 -> [1]
	"""
	# sd == 1 -> 8 + 8 -> 16
	# sd == 0 -> 8 + 0 -> 8
	return 8 + (sd << 3)

def opp_side_tag(sd):
	"""
	判断边界标签(16)，获取对方红黑标记
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	# sd == 1 -> 16 - 8
	# sd == 0 -> 16 - 0
	return 16 - (sd << 3)

def src(mv):
	"""
	原始位置，获得走法的起点
	"""
	return mv & 255

def dst(mv):
	"""
	目标位置，获得走法的终点
	"""
	return mv >> 8

def move(sq_src, sq_dst):
	"""
	移动位置，根据起点和终点获得走法，返回计算型Move
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	# 左移8位，产生前面的格子(<<8)
	return sq_src + (sq_dst << 8)

def mirror_move(mv):
	"""
	todo: 对称局面移动，走法水平镜像
	"""
	return move(mirror_squares(src(mv)), mirror_squares(dst(mv)))

def mvv_lva(pc, lva):
	"""
	todo: MVV-LVA排序
	MVV/LVA是静态搜索最常用的启发方式，如果棋盘的数据结构精心设计，SEE也是值得尝试的。
	尽管MVV/LVA很简单，但是究竟根据“被吃子价值-攻击子价值”来排序，还是先排序“被吃子价值”
	相同的情况再排序“攻击子价值”呢？ElephantEye是以MVV(LVA)的值为依据

	MVV/LVA意思就是最有价值的受害者/最没价值的攻击者[Most Valuable Victim/Least Valuable Attacker]
	从而最先搜索最好的吃子着法，这个计数假设最好的吃子是吃到最大的子，如果不止一个棋子能够吃到最大的子，那么假设
	用最小的子去吃掉最大的子，可以解决静态搜索膨胀的问题
	"""
	# pc: 每个棋子的标识
	return MVV_VALUE[pc & 7] - lva

def new_chr(n):
	"""
	返回值是当前整数的ASCII字符，用一个在range(0, 255)整数作为参考，返回对应字符
	该函数的返回值为字符串形式，例如，输入: chr(90)，输出为: ‘Z’
	"""
	return chr(n)

def asc(c):
	"""
	与chr()函数对应，输入ASCII字符表中字符的字符串形式
	返回在字符表的排序位次，例如: ord('Z')，输出为: 90
	"""
	return ord(c)

def char_to_piece(c):
	"""
	todo: 字符与棋子解析转换[匹配棋子]
	"""
	if c == "K":
		return PIECE_KING  # 将/帅
	elif c == "A":
		return PIECE_ADVISOR  # 士/仕
	elif c == "B":
		return PIECE_BISHOP  # 相
	elif c == "E":
		return PIECE_BISHOP  # 象
	elif c == "H":
		return PIECE_KNIGHT  # 马
	elif c == "N":
		return PIECE_KNIGHT  # 傌
	elif c == "R":
		return PIECE_ROOK  # 车
	elif c == "C":
		return PIECE_CANNON  # 炮
	elif c == "P":
		return PIECE_PAWN  # 兵/卒
	else:
		return -1

def un_singed_right_shift(x, y):
	"""
	无符号右移
	todo: 运算
		^ 两个位相同为0，相异为1
		& 两个位都为1时，结果才为1
		| 两个位都为0时，结果才为0
		~ 0变1，1变0
		>> 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
		<< 各二进位全部左移若干位，高位丢弃，低位补0
	"""
	x = x & 0xffffffff  # 0xffffffff -> 4294967295
	signed = False  # 表示需要考虑符号位
	if x < 0:
		signed = True
	x = x.to_bytes(4, byteorder='big', signed=signed)  # 十进制转换为bytes
	x = int.from_bytes(x, byteorder='big', signed=False)  # bytes转换为十进制
	return x >> (y & 0xf)  # 0xf -> 15