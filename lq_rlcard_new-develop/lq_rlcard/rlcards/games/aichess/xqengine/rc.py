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
通常把走法赋值给mv_result，并把mv_last % 256和mv_last / 256这两个格子进行标记
"""


class RC4:
	"""
	todo: 设计出的密钥长度可变的加密算法簇
	RC4密码流生成器
	"""
	def __init__(self, key):
		"""
		初始化棋盘信息
		"""
		self.x = self.y = 0
		self.state = list(range(0, 256))
		flag_idx = 0
		# 用空密匙初始化密码流生成器
		for init_idx in range(256):
			flag_idx = (flag_idx + self.state[init_idx] + key[init_idx % len(key)]) & 0xff  # 0xff -> 255
			self.swap(init_idx, flag_idx)

	def swap(self, start_idx, end_idx):
		"""
		交换
		"""
		self.state[start_idx], self.state[end_idx] = self.state[end_idx], self.state[start_idx]

	def next_byte(self):
		"""
		生成密码流的下一个字节
		"""
		self.x = (self.x + 1) & 0xff  # 0xff -> 255
		self.y = (self.y + self.state[self.x]) & 0xff
		self.swap(self.x, self.y)
		t = (self.state[self.x] + self.state[self.y]) & 0xff
		return self.state[t]

	def next_long(self):
		"""
		生成密码流的下四个字节
		"""
		n0 = self.next_byte()
		n1 = self.next_byte()
		n2 = self.next_byte()
		n3 = self.next_byte()
		return ((n0 + (n1 << 8) + (n2 << 16) + ((n3 << 24) & 0xffffffff)) + 2147483648) % 4294967296 - 2147483648


pre_gen_zob_key_table = []
pre_gen_zob_lock_table = []

rc4 = RC4([0])
pre_gen_zob_key_player = rc4.next_long()
rc4.next_long()
pre_gen_zob_lock_player = rc4.next_long()

for i in range(14):
	keys = []
	locks = []
	for j in range(256):
		keys.append(rc4.next_long())
		rc4.next_long()
		locks.append(rc4.next_long())
	pre_gen_zob_key_table.append(keys)
	pre_gen_zob_lock_table.append(locks)