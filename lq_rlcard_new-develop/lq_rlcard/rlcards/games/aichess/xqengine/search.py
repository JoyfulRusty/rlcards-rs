# -*- coding: utf-8 -*-

import math
import time
import random
import concurrent.futures

from rlcards.games.aichess.xqengine.util import shell_sort, home_half, dst
from rlcards.games.aichess.xqengine.const import MATE_VALUE, WIN_VALUE, BAN_VALUE

"""
P -> 兵
C -> 炮
R -> 车
N -> 马
B -> 象
Q -> 士
K -> 帅

1.杀手走法就是兄弟节点中产生的beta截断走法，根据国际象棋的经验，杀手走法产生截断的可能性极大，所以在中国象棋中吸取了这个经验，很显然，兄弟节点的走法未
必在当下节点下

2.在尝试杀手走法以前先要对它进行走法合理性的判断，legal_move函数的作用就是，如果杀手走法确实产生了截断了，那么后面耗时更多的generate_move将停止执行

3.如何保存和获取“兄弟节点中产生截断的走法”呢？把这个问题简单化——距离根节，点步数(nDistance)同样多的节点，彼此都称为“兄弟”节点，换句话说，亲兄弟、
堂表兄弟以及关系更疏远的兄弟都称为“兄弟”，可以把距离根节点的步数(nDistance)作为索引值，构造一个杀手走法表

象棋的每个杀手走法表项存有两个杀手走法，走法一比走法二优先: 存一个走法时，走法二被走法一替换，走法一被新走法替换；取走法时，先取走法一，后取走法二
"""

PHASE_HASH = 0  # 哈希
PHASE_KILLER_1 = 1  # 杀手走法一
PHASE_KILLER_2 = 2  # 杀手走法二
PHASE_GEN_MOVES = 3  # 最大的生成走法数
PHASE_REST = 4  # 重置

LIMIT_DEPTH = 64  # 搜索的极限深度
NULL_DEPTH = 2  # 空着裁剪的深度
RANDOMNESS = 8  # 随机性

HASH_ALPHA = 1  # 哈希α，ALPHA节点的置换表项
HASH_BETA = 2  # 哈希β，BETA节点的置换表项
HASH_PV = 3  # 哈希PV，PV节点的置换表项


class HashTable:
	"""
	哈希表
	"""
	def __init__(self, depth, flag, vl, mv, zob_lock):
		"""
		初始化参数
		"""
		self.vl = vl  # 分值
		self.mv = mv  # 最佳走法
		self.flag = flag  # 标志位
		self.depth = depth  # 搜索深度
		self.zob_lock = zob_lock  # 校验锁🔒

class MoveSort:
	"""
	动作排序
	"""
	def __init__(self, mv_hash, pos, killer_table, history_table):
		"""
		初始化参数
		"""
		self.mvs = []
		self.vls = []
		self.pos = pos
		self.index = 0
		self.mv_hash = 0
		self.mv_killer1 = 0
		self.mv_killer2 = 0
		self.single_reply = False
		self.phase = PHASE_HASH  # 0，阶段/移动相位
		self.history_table = history_table

		# todo: 检查
		if pos.in_check():
			self.phase = PHASE_REST  # 4
			all_mvs = pos.generate_moves(None)
			for i in range(len(all_mvs)):
				mv = all_mvs[i]
				if not pos.make_move(mv):
					continue
				pos.undo_make_move()
				self.mvs.append(mv)
				# 0x7fffffff -> 2147483647
				self.vls.append(0x7fffffff if mv == mv_hash else history_table[pos.history_index(mv)])
			shell_sort(self.mvs, self.vls)
			self.single_reply = len(self.mvs) == 1
		else:
			# todo: 初始化杀手走法
			self.mv_hash = mv_hash
			self.mv_killer1 = killer_table[pos.distance][0]
			self.mv_killer2 = killer_table[pos.distance][1]

	def next(self):
		"""
		todo: 优化走法，根据着法排序策略得到下一个着法
			[置换表走法 -> 杀手走法一 -> 杀手走法二 -> 生成所有走法(循环) -> 选出最佳走法]
			利用各种信息渠道(如置换表、杀手走法、历史表等)来对走法进行优化:
				1.如果是置换表中存在过该局面的局面，但无法完全利用，那么多数情况下，它是浅一层搜索中产生截断的走法
				2.然后，两个杀手走法，如果其中某个杀手走法与置换表走法一样，那么可跳过
				3.然后，生成全部走法，按历史表排序，再依次搜索，可排序置换表和两个杀手走法一样，构造状态机，描述走法顺序若干阶段

		先判断杀手着法的合理性，判断着法合理性花费的时间比产生全部着法少的多，当存在合法着法时则先搜索此类着法，因为杀手着法是产生截断机率很高的着法
		"""
		# todo: 置换表启发
		if self.phase == PHASE_HASH:
			self.phase = PHASE_KILLER_1
			if self.mv_hash > 0:
				return self.mv_hash

		# todo: 杀手走法一(杀手节点1)
		if self.phase == PHASE_KILLER_1:
			self.phase = PHASE_KILLER_2
			if self.mv_killer1 != self.mv_hash and self.mv_killer1 > 0 and self.pos.legal_move(self.mv_killer1):
				return self.mv_killer1

		# todo: 杀手走法二(杀手节点2)
		if self.phase == PHASE_KILLER_2:
			self.phase = PHASE_GEN_MOVES
			if self.mv_killer2 != self.mv_hash and self.mv_killer2 > 0 and self.pos.legal_move(self.mv_killer2):
				return self.mv_killer2

		# todo: 生成着法动作并按照历史表排序
		if self.phase == PHASE_GEN_MOVES:
			self.phase = PHASE_REST
			self.mvs = self.pos.generate_moves(None)
			self.vls = []
			# 置换表
			for i in range(len(self.mvs)):
				self.vls.append(self.history_table[self.pos.history_index(self.mvs[i])])
			shell_sort(self.mvs, self.vls)
			self.index = 0

		# 对于剩余的非置换表，非杀手节点逐个获取
		while self.index < len(self.mvs):
			mv = self.mvs[self.index]
			self.index += 1
			if mv != self.mv_hash and mv != self.mv_killer1 and mv != self.mv_killer2:
				return mv
		# 着法取尽，返回0
		return 0


class Search:
	"""
	搜索类
	"""
	def __init__(self, pos, hash_level):
		"""
		初始化参数
		"""
		self.pos = pos
		self.mv_result = 0
		self.all_nodes = 0
		self.all_millis = 0
		self.hash_table = []  # 置换表走法
		self.killer_table = []  # 杀手走法表
		self.history_table = []  # 历史表
		self.hash_mask = (1 << hash_level) - 1

	def get_hash_item(self):
		"""
		获取历史表中哈希元素
		"""
		hash_value = self.pos.zob_key & self.hash_mask
		return self.hash_table[hash_value]

	def probe_hash(self, vl_alpha, vl_beta, depth, mv):
		"""
		todo: probe_hash -> 利用置换表信息
			1.检查局面所对应的置换表项，如果与z_lock校验码匹配，那么则认为命中[hit]
			2.是否能直接利用置换表中的结果，取决于两个因素:
				A: 深度是否达到要求
				B：非PV节点是否考虑边界
		"""
		hash_table = self.get_hash_item()
		if hash_table.zob_lock != self.pos.zob_lock:
			mv[0] = 0
			return -MATE_VALUE

		mv[0] = hash_table.mv
		mate = False  # 是否被将
		if hash_table.vl > WIN_VALUE:
			if hash_table.vl <= BAN_VALUE:
				return -MATE_VALUE
			hash_table.vl -= self.pos.distance
			mate = True

		elif hash_table.vl < -WIN_VALUE:
			if hash_table.vl >= -BAN_VALUE:
				return -MATE_VALUE
			hash_table.vl += self.pos.distance
			# 杀棋标志：如果是杀棋，那么不需要满足深度条件
			mate = True

		# todo: 和棋
		elif hash_table.vl == self.pos.draw_value():
			return -MATE_VALUE

		if hash_table.depth < depth and not mate:
			return -MATE_VALUE
		if hash_table.flag == HASH_BETA:
			return hash_table.vl if hash_table.vl >= vl_beta else -MATE_VALUE
		if hash_table.flag == HASH_ALPHA:
			return hash_table.vl if hash_table.vl <= vl_alpha else -MATE_VALUE

		return hash_table.vl

	def record_hash(self, flag, vl, depth, mv):
		"""
		todo: 记录哈希值、置换表
			1.没有置换表，则称不上完成的计算机博弈程序，置换表非常简单，以z_key % HASH_SIZE作为索引
			2.每个置换表项存储的内容无非就是:
				A. 深度
				B. 标志
				C. 分值
				D. 最佳走法
				E. zob_lock校验码
			3.record_hash即采用深度优先的替换策略，在判断深度后，将HASH表项中的每一个值填上即可
			4.probe_hash返回一个非 -MATE_VALUE的值，这样就能不对该节点进行展开，如果仅仅符合第一中情况，那么该置换表项的信息仍旧有意义，它的
			最佳走法给了一定的启发[部分利用]

		从学会走棋开始，就开始考虑了杀棋分数，不过增加置换表以后，这个分数需要进行调整: 置换表中的分值不能是距离根节点的杀棋分值，而是距离当前置换表
		项节点的分值，所以当分值接近INFINITY或-INFINITY时，probe_hash和record_hash都要做细微的调整:
			1.对于record_hash: 置换表项记录的杀棋步数 = 实际杀棋步数 - 置换表项距离根节点的步数
			2.对于probe_hash: 实际杀棋步数 = 置换表项记录的杀棋步数 + 置换表项距离根节点的步数
		"""
		hash_table = self.get_hash_item()
		if hash_table.depth > depth:
			return
		hash_table.flag = flag
		hash_table.depth = depth
		if vl > WIN_VALUE:
			if mv == 0 and vl <= BAN_VALUE:
				return
			hash_table.vl = vl + self.pos.distance
		elif vl < -WIN_VALUE:
			if mv == 0 and vl >= -BAN_VALUE:
				return
			hash_table.vl = vl - self.pos.distance
		# 判断和棋
		elif vl == self.pos.draw_value() and mv == 0:
			return
		else:
			hash_table.vl = vl
		hash_table.mv = mv
		hash_table.zob_lock = self.pos.zob_lock

	def set_best_move(self, mv_result, depth):
		"""
		最佳走法的处理，设置最佳走法
		"""
		self.history_table[self.pos.history_index(mv_result)] += depth * depth
		mvs_killer = self.killer_table[self.pos.distance]
		if mvs_killer[0] != mv_result:
			mvs_killer[1] = mvs_killer[0]
			mvs_killer[0] = mv_result

	def search_quiescent(self, vl_alpha, vl_beta):
		"""
		todo: [只考虑吃子着法]静态搜索时，分两种情况
			1.不被将军，首先尝试不走，是否能被截断，然后搜索所有吃子的走法(可按照MVV或LVA排序)
			2.被将军，这时必须生成所有走法，可按照历史表排序

		搜索静止，克服水平线效应的方法:
		(1).静态搜索(Quiescence)
			1.不被将军，首先尝试不走，是否能被截断，然后搜索所有吃子的走法(可按照MVV或LVA排序)
			2.被将军，这时必须生成所有走法，可按照历史表排序
		(2).空步裁剪(NullMove)，某些条件下并不适用
			1.被将军的情况下
			2.进入残局时(自己一方的子力总价值小于某个阈值)
			3.不要连续做两次空步裁剪，否则会导致搜索的退化
		(3).将军延申

		静态搜索思想: 达到主搜索的水平线后，用一个图灵型的搜索只展开吃子(有时是吃子加将军)的着法，静态搜索还必须包包括放弃的着法，避免了在明显有对策
		的情况下看错局势，简而言之，静态搜索就是应对可能的动态局面的搜索
		"""
		vl_alpha = vl_alpha
		self.all_nodes += 1
		vl = self.pos.mate_value()
		if vl >= vl_beta:
			return vl
		vl_rep = self.pos.rep_status(1)
		if vl_rep > 0:
			return self.pos.rep_value(vl_rep)
		# 达到限制深度，则进行评估
		if self.pos.distance == LIMIT_DEPTH:
			return self.pos.evaluate()
		vl_best = -MATE_VALUE
		vls = []
		# 处在被将军局面，生成所有着法
		if self.pos.in_check():
			mvs = self.pos.generate_moves(None)
			for i in range(len(mvs)):
				vls.append(self.history_table[self.pos.history_index(mvs[i])])
			shell_sort(mvs, vls)
		else:
			# 调用静态评价，如果评价好得足以截断而不需要试图吃子时，马上截断，返回beta，如果评价不足以产生截断，但是比alpha好，那么就更新alpha来
			# 反映静态评价，然后尝试吃子着法，如果其中任何一个产生截断，搜索就终止，可能它们没有一个是好的，但不存在什么问题。可能评价函数会返回足够
			# 高的数值，使得函数通过beta截断马上返回，也可能某个吃子产生beta截断，可能静态评价比较坏，而任何吃子着法也不会更好，或者可能任何吃子都
			# 不好，但是静态搜索只比alpha高一点点
			# todo: 未被将军，先对局面进行评价
			vl = self.pos.evaluate()
			if vl > vl_best:
				if vl >= vl_beta:
					return vl
				vl_best = vl
				vl_alpha = max(vl, vl_alpha)
			# 再使用MVV-LVA启发对着法排序
			mvs = self.pos.generate_moves(vls)
			shell_sort(mvs, vls)
			for i in range(len(vls)):
				if vls[i] < 10 or (vls[i] < 20 and home_half(dst(mvs[i]), self.pos.sd_player)):
					mvs = mvs[:i]
					break
		for i in range(len(mvs)):
			# 判断是否都是合法的着法
			if not self.pos.make_move(mvs[i]):
				continue
			vl = -self.search_quiescent(-vl_beta, -vl_alpha)
			# 撤销着法
			self.pos.undo_make_move()
			# 用Alpha-Beta算法搜索
			if vl > vl_best:
				if vl >= vl_beta:
					return vl
				vl_best = vl
				# alpha为搜索到的最好值，任何比它更小的值就无用，因为策略就是知道alpha的值，任何小于或等于alpha的值都不会有所提高
				# beta为对于对手来说的最坏的值，这是对手所能承受的最坏结果，在对手看来，它总会找到一个对策不比beta更坏的
				vl_alpha = max(vl, vl_alpha)
		return self.pos.mate_value() if vl_best == -MATE_VALUE else vl_best

	def search_full(self, vl_alpha, vl_beta, depth, no_null):
		"""
		完全搜索
		"""
		vl_alpha = vl_alpha
		# 对叶子节点使用静态搜索
		if depth <= 0:
			return self.search_quiescent(vl_alpha, vl_beta)
		self.all_nodes += 1
		vl = self.pos.mate_value()
		if vl >= vl_beta:
			return vl
		# 重复裁剪
		vl_rep = self.pos.rep_status(1)
		if vl_rep > 0:
			return self.pos.rep_value(vl_rep)

		mv_hash = [0]
		vl = self.probe_hash(vl_alpha, vl_beta, depth, mv_hash)
		if vl > -MATE_VALUE:
			return vl
		# 搜索达到极限深度，返回评价值
		if self.pos.distance == LIMIT_DEPTH:
			return self.pos.evaluate()
		# 尝试空着裁剪
		if not no_null and not self.pos.in_check() and self.pos.null_okay():
			self.pos.null_move()
			# 空着裁剪安全，记录深度至少为NULL_DEPTH+1
			vl = -self.search_full(-vl_beta, 1 - vl_beta, depth - NULL_DEPTH - 1, True)
			self.pos.undo_null_move()
			# 空着裁剪安全，记录深度至少为NULL_DEPTH
			if vl >= vl_beta and (
					self.pos.null_safe() or self.search_full(vl_alpha, vl_beta, depth - NULL_DEPTH, True) >= vl_beta):
				return vl

		hash_flag = HASH_ALPHA
		vl_best = -MATE_VALUE
		mv_best = 0
		sort = MoveSort(mv_hash[0], self.pos, self.killer_table, self.history_table)
		while True:
			mv = sort.next()
			if mv <= 0:
				break
			if not self.pos.make_move(mv):
				continue
			new_depth = depth if (self.pos.in_check() or sort.single_reply) else depth - 1
			if vl_best == -MATE_VALUE:
				vl = -self.search_full(-vl_beta, -vl_alpha, new_depth, False)
			else:
				vl = -self.search_full(-vl_alpha - 1, vl_alpha, new_depth, False)
				if vl_alpha < vl < vl_beta:
					vl = -self.search_full(-vl_beta, -vl_alpha, new_depth, False)
			self.pos.undo_make_move()
			# todo: 更新最好的移动动作，使用Alpha-Beta算法搜索
			if vl > vl_best:
				vl_best = vl
				if vl >= vl_beta:
					hash_flag = HASH_BETA
					mv_best = mv
					break
				if vl > vl_alpha:
					vl_alpha = vl
					hash_flag = HASH_PV
					mv_best = mv
		# 判断是否被将死
		if vl_best == -MATE_VALUE:
			return self.pos.mate_value()
		# 采用深度优先的替换策略，在判断深度后，将HASH表项中的每一个值填上即可
		# 置换表项记录的杀棋步数 = 实际杀棋步数 - 置换表项距离根节点的步数
		# 更新置换表、历史表和杀手着法表
		self.record_hash(hash_flag, vl_best, depth, mv_best)
		if mv_best > 0:
			# 更新杀手走法表中最好的动作
			self.set_best_move(mv_best, depth)
		return vl_best

	def search_root(self, depth):
		"""
		搜索根节点
		"""
		vl_best = -MATE_VALUE
		sort = MoveSort(self.mv_result, self.pos, self.killer_table, self.history_table)
		while True:
			mv = sort.next()
			if mv <= 0:
				break
			if not self.pos.make_move(mv):
				continue
			# 选择性延申
			new_depth = depth if self.pos.in_check() else depth - 1
			# 主要变列搜索
			if vl_best == -MATE_VALUE:
				vl = -self.search_full(-MATE_VALUE, MATE_VALUE, new_depth, True)
			else:
				vl = -self.search_full(-vl_best - 1, -vl_best, new_depth, False)
				if vl > vl_best:
					vl = -self.search_full(-MATE_VALUE, -vl_best, new_depth, True)
			self.pos.undo_make_move()
			if vl > vl_best:
				# 搜索到最佳着法时，记录主要变列
				vl_best = vl
				self.mv_result = mv
				# print("输出当前更新的mv_result: ", self.mv_result)
				if -WIN_VALUE < vl_best < WIN_VALUE:
					vl_best += math.floor(random.random() * RANDOMNESS) - math.floor(random.random() * RANDOMNESS)
					vl_best = ((vl_best - 1) if vl_best == self.pos.draw_value() else vl_best)
		self.set_best_move(self.mv_result, depth)
		return vl_best

	def search_unique(self, vl_beta, depth):
		"""
		搜索唯一
		"""
		sort = MoveSort(self.mv_result, self.pos, self.killer_table, self.history_table)
		sort.next()
		while True:
			mv = sort.next()
			if mv <= 0:
				break
			if not self.pos.make_move(mv):
				continue
			vl = -self.search_full(-vl_beta, 1 - vl_beta, depth if self.pos.in_check() else depth - 1, False)
			self.pos.undo_make_move()
			if vl >= vl_beta:
				return False
		return True

	def search_main(self, depth, millis):
		"""
		todo: 启动搜索着法生成流程
		"""
		# 开局读取开局库中的着法
		self.mv_result = self.pos.book_move()
		if self.mv_result > 0:
			# 当开局库中的着法构成循环局面，那么不走这个着法
			self.pos.make_move(self.mv_result)
			if self.pos.rep_status(3) == 0:
				self.pos.undo_make_move()
				return self.mv_result
			# 撤销着法
			self.pos.undo_make_move()

		# todo: 清空哈希表
		self.hash_table = []
		self.hash_table.extend([HashTable(depth=0, flag=0, vl=0, mv=0, zob_lock=0)] * (self.hash_mask + 1))

		# todo: 清空杀手走法表
		self.killer_table = []
		self.killer_table.extend([[0, 0]] * LIMIT_DEPTH)

		# todo: 清空历史表
		self.history_table = []
		self.history_table.extend([0] * 4096)

		self.mv_result = 0
		self.all_nodes = 0
		self.pos.distance = 0

		# todo: 剪枝搜索
		start_time = time.time()
		for i in range(1, depth + 1):
			vl = self.search_root(i)
			cost_time = time.time() - start_time
			print(f"搜索第{i}次，耗费时间为: {cost_time}")
			self.all_millis = (time.time() - start_time) * 500
			if self.all_millis > millis:
				break
			if vl > WIN_VALUE or vl < -WIN_VALUE:
				break
			if self.search_unique(1 - WIN_VALUE, i):
				break
		return self.mv_result