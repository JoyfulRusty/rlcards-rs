# -*- coding: utf-8 -*-

from rlcards.games.aichess.alphabeta import const

from board import ChessBoard
from relation import Relation
from history import HistoryCache
from rlcards.games.aichess.chess.step import Step
from rlcards.games.aichess.chess.chess import Chess


class AlphaBeta:
	"""
	alpha-beta剪枝算法
	"""
	def __init__(self):
		"""
		初始化剪枝属性参数
		"""
		self.board = ChessBoard()  # 象棋棋盘
		self.max_depth = const.max_depth  # 搜索深度[也就是能看到的步数]
		self.history_table = HistoryCache()  # 启发式缓存历史数据表
		self.best_move = Step()  # 更新最好move
		self.cnt = 0

	def alpha_by_beta(self, depth, alpha, beta):
		"""
		alpha-beta剪枝，alpha是大可能下界，beta是最小可能上界
		"""
		who = (self.max_depth - depth) % 2  # 那个玩家
		# 判断游戏是否结束，游戏结束则不再进行搜索
		if self.is_game_over(who):
			return const.min_val
		# 搜索到指定深度，则不再进行搜索
		# 评估搜索结果
		# 搜索到叶子节点
		if depth == 1:
			return self.evaluate(who)
		# 生成能走的合法路线
		move_list = self.board.generate_move(who)
		# 利用历史表0
		for i in range(len(move_list)):
			move_list[i].score = self.history_table.get_history_score(who, move_list[i])
		# 为了更容易剪枝利用历史表得分进行排序
		move_list.sort()
		best_step = move_list[0]
		# 记录分数
		score_list = []
		for step in move_list:
			# 更新step记录
			temp = self.move_to(step)
			# 因为一层选最大，一层选最小，利用取负好来实现
			score = -self.alpha_by_beta(depth - 1, -beta, -alpha)
			# 添加分数
			score_list.append(score)
			# 恢复动作移动
			self.undo_move(step, temp)
			# 根据分数来选取最好的动作
			if score > alpha:
				alpha = score
				if depth == self.max_depth:
					self.best_move = step
				# 进行剪枝操作替换
				best_step = step
			# α>β时，则直接跳出，已获取到最佳动作更新
			if alpha >= beta:
				best_step = step
				break
		# 更新历史表
		if best_step.from_x != -1:
			self.history_table.add_history_score(who, best_step, depth)
		return alpha

	def evaluate(self, who):
		"""
		评估
		who表示该谁走，返回其评分值
		"""
		self.cnt += 1
		relation_list = self.init_relation_list()
		tmp_base_val = [0, 0]
		tmp_pos_val = [0, 0]
		tmp_mobile_val = [0, 0]
		tmp_relation_val = [0, 0]
		for x in range(9):
			for y in range(10):
				now_chess = self.board.board[x][y]
				now_chess_type = now_chess.chess_type
				if now_chess_type == 0:
					continue
				# now_belong = 0 if who else 1
				now_belong = now_chess.belong
				pos = x * 9 + y
				temp_move_list = self.board.get_chess_move(x, y, now_belong, True)
				# 计算基础价值
				tmp_base_val[now_belong] += const.base_val[now_chess_type]
				# 计算位置价值
				if now_belong == 0:
					# 当要求最大值玩家
					tmp_pos_val[now_belong] += const.pos_val[now_chess_type][pos]
				else:
					tmp_pos_val[now_belong] += const.pos_val[now_chess_type][89 - pos]
				# 计算机动性价值，记录关系信息
				for item in temp_move_list:
					# print('----------------')
					# print(item)
					# 目的位置棋子
					temp_chess = self.board.board[item.to_x][item.to_y]
					# 为空时，那么加上机动性值
					if temp_chess.chess_type == const.kong:
						tmp_mobile_val[now_belong] += const.mobile_val[now_chess_type]
						# print(tmp_mobile_val[now])
						continue
					# 当不是自己一方棋子时
					elif temp_chess.belong != now_belong:
						if temp_chess.chess_type == const.jiang:  # 如果能吃了对方的将，那么就赢了
							if temp_chess.belong != who:
								return const.max_val
							else:
								tmp_relation_val[1 - now_belong] -= 20  # 如果不能，那么就相当于被将军，对方要减分
								continue
						# 记录攻击了谁
						relation_list[x][y].attack[relation_list[x][y].num_attack] = temp_chess.chess_type
						relation_list[x][y].num_attack += 1
						relation_list[item.to_x][item.to_y].chess_type = temp_chess.chess_type
						# 记录被谁攻击
						relation_list[item.to_x][item.to_y].attacked[
							relation_list[item.to_x][item.to_y].num_attacked] = now_chess_type
						relation_list[item.to_x][item.to_y].num_attacked += 1

					elif temp_chess.belong == now_belong:
						if temp_chess.chess_type == const.jiang:  # 保护自己的将没有意义，直接跳过
							continue
						# 记录关系信息 - guard
						relation_list[x][y].guard[relation_list[x][y].num_guard] = temp_chess
						relation_list[x][y].num_guard += 1
						relation_list[item.to_x][item.to_y].chess_type = temp_chess.chess_type

						relation_list[item.to_x][item.to_y].guarded[
							relation_list[item.to_x][item.to_y].num_guarded] = now_chess_type
						relation_list[item.to_x][item.to_y].num_guarded += 1
		for x in range(9):
			for y in range(10):
				num_attacked = relation_list[x][y].num_attacked
				num_guarded = relation_list[x][y].num_guarded
				now_chess = self.board.board[x][y]
				now_chess_type = now_chess.chess_type
				now_belong = now_chess.belong
				unit_val = const.base_val[now_chess.chess_type] >> 3
				sum_attack = 0  # 被攻击总子力
				sum_guard = 0
				min_attack = 999  # 最小的攻击者
				max_attack = 0  # 最大的攻击者
				max_guard = 0
				flag = 999  # 有没有比这个子的子力小的
				if now_chess_type == const.kong:
					continue
				# 统计攻击方的子力
				for i in range(num_attacked):
					temp = const.base_val[relation_list[x][y].attacked[i]]
					flag = min(flag, min(temp, const.base_val[now_chess_type]))
					min_attack = min(min_attack, temp)
					max_attack = max(max_attack, temp)
					sum_attack += temp
				# 统计防守方的子力
				for i in range(num_guarded):
					temp = const.base_val[relation_list[x][y].guarded[i]]
					max_guard = max(max_guard, temp)
					sum_guard += temp
				if num_attacked == 0:
					tmp_relation_val[now_belong] += 5 * relation_list[x][y].num_guarded
				else:
					mut_val = 5 if who != now_belong else 1
					if num_guarded == 0:  # 如果没有保护
						tmp_relation_val[now_belong] -= mut_val * unit_val
					else:  # 如果有保护
						if flag != 999:  # 存在攻击者子力小于被攻击者子力,对方将愿意换子
							tmp_relation_val[now_belong] -= mut_val * unit_val
							tmp_relation_val[1 - now_belong] -= mut_val * (flag >> 3)
						# 如果是二换一, 并且最小子力小于被攻击者子力与保护者子力之和, 则对方可能以一子换两子
						elif num_guarded == 1 and num_attacked > 1 and min_attack < const.base_val[now_chess_type] + sum_guard:
							tmp_relation_val[now_belong] -= mut_val * unit_val
							tmp_relation_val[now_belong] -= mut_val * (sum_guard >> 3)
							tmp_relation_val[1 - now_belong] -= mut_val * (flag >> 3)
						# 如果是三换二并且攻击者子力较小的二者之和小于被攻击者子力与保护者子力之和,则对方可能以两子换三子
						elif num_guarded == 2 and num_attacked == 3 and sum_attack - max_attack < const.base_val[now_chess_type] + sum_guard:
							tmp_relation_val[now_belong] -= mut_val * unit_val
							tmp_relation_val[now_belong] -= mut_val * (sum_guard >> 3)
							tmp_relation_val[1 - now_belong] -= mut_val * ((sum_attack - max_attack) >> 3)
						# 如果是n换n，攻击方与保护方数量相同并且攻击者子力小于被攻击者子力与保护者子力之和再减去保护者中最大子力,则对方可能以n子换n子
						elif num_guarded == num_attacked and sum_attack < const.base_val[now_chess.chess_type] + sum_guard - max_guard:
							tmp_relation_val[now_belong] -= mut_val * unit_val
							tmp_relation_val[now_belong] -= mut_val * ((sum_guard - max_guard) >> 3)
							tmp_relation_val[1 - now_belong] -= sum_attack >> 3
		my_max_val = tmp_base_val[0] + tmp_pos_val[0] + tmp_mobile_val[0] + tmp_relation_val[0]
		my_min_val = tmp_base_val[1] + tmp_pos_val[1] + tmp_mobile_val[1] + tmp_relation_val[1]
		if who == 0:
			return my_max_val - my_min_val
		else:
			return my_min_val - my_max_val

	@staticmethod
	def init_relation_list():
		"""
		初始化关系列表
		"""
		res_list = []
		for i in range(9):
			res_list.append([])
			for j in range(10):
				res_list[i].append(Relation())
		return res_list

	def is_game_over(self, who):
		"""
		判断游戏是否结束
		"""
		for i in range(9):
			for j in range(10):
				# 判断当前棋是否为将/帅
				if self.board.board[i][j].chess_type == const.jiang:
					# 判断当前的棋属于那位玩家
					if self.board.board[i][j].belong == who:
						# 返回False
						return False
		return True

	def move_to(self, step):
		"""
		计算移动棋子的玩家和棋子
		"""
		belong = self.board.board[step.to_x][step.to_y].belong
		chess_type = self.board.board[step.to_x][step.to_y].chess_type
		temp = Chess(belong, chess_type)
		# 更新棋子的移动chess_type
		self.board.board[step.to_x][step.to_y].chess_type = self.board.board[step.from_x][step.from_y].chess_type
		self.board.board[step.to_x][step.to_y].belong = self.board.board[step.from_x][step.from_y].belong
		# 重置动作位置
		self.board.board[step.from_x][step.from_y].chess_type = const.kong
		self.board.board[step.from_x][step.from_y].belong = -1
		return temp

	def undo_move(self, step, chess):
		"""
		恢复棋子
		"""
		# 恢复棋子的移动chess_type
		self.board.board[step.from_x][step.from_y].belong = self.board.board[step.to_x][step.to_y].belong
		self.board.board[step.from_x][step.from_y].chess_type = self.board.board[step.to_x][step.to_y].chess_type
		self.board.board[step.to_x][step.to_y].belong = chess.belong
		self.board.board[step.to_x][step.to_y].chess_type = chess.chess_type

if __name__ == "__main__":
	game = AlphaBeta()
	while True:
		from_x = int(input())
		from_y = int(input())
		to_x = int(input())
		to_y = int(input())
		s = Step(from_x, from_y, to_x, to_y)
		game.alpha_by_beta(game.max_depth, const.min_val, const.max_val)
		print("输出最好的移动: ", game.best_move)
		game.move_to(game.best_move)