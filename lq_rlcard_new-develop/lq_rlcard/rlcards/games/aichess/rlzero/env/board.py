# -*- coding: utf-8 -*-

import copy
import numpy as np

from rlcards.games.aichess.rlzero.const import const
from rlcards.games.aichess.rlzero.config import CONFIG
from rlcards.games.aichess.rlzero.util.util import \
	get_legal_moves, \
	state_list2state_array, \
	move_action2move_id, \
	move_id2move_action


class Board:
	"""
	棋盘
	"""
	def __init__(self):
		"""
		初始化棋盘参数
		"""
		self.winner = None
		self.kill_action = 0
		self.last_move = None
		self.game_start = False
		self.have_pos = copy.deepcopy(const.STATE_GUY_INIT)
		self.state_list = copy.deepcopy(const.STATE_LIST_INIT)
		self.state_deque = copy.deepcopy(const.STATE_QUEUE_INIT)

		self.start_player = 1
		self.backhand_player = 2
		self.id2color = {1: '红', 2: '黑'}
		self.color2id = {'红': 1, '黑': 2}

		self.curr_player_color = self.id2color[self.start_player]  # 红
		self.curr_player_id = self.color2id['红']
		self.action_count = 0

	def init_board(self):
		"""
		初始化棋盘方法
		传入先手玩家ID
		"""
		# 当前手玩家，也就是先手玩家
		self.curr_player_color = self.id2color[self.start_player]  # 红
		self.curr_player_id = self.color2id['红']

		# 初始化棋盘状态和状态队列
		self.state_list = copy.deepcopy(const.STATE_LIST_INIT)
		self.state_deque = copy.deepcopy(const.STATE_QUEUE_INIT)

		# 记录游戏中吃子的回合数
		self.kill_action = 0
		self.game_start = False
		self.action_count = 0  # 游戏动作计数器

	@property
	def legal_moves(self):
		"""
		获的当前盘面的所有合法走子集合
		"""
		moves = get_legal_moves(self.state_deque, self.curr_player_color)
		return moves

	def curr_map(self):
		"""
		当前map
		"""
		return self.state_deque[-1], self.curr_player_color

	def get_curr_state(self):
		"""
		从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9] 走一步 CHW

		todo: 解释[9, 10, 9]
			1.使用9个平面来表示棋盘状态
			2.[0-6]平面表示棋子位置，1代表红方棋子，-1代表黑方棋子，队列最后一个平面
			3.[7]个平面表示对手player上一个落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部表示为0
			4.[8]个平面表示当前player是不是先手player，如果为先手player则整个平面全部表示为1，否则全部表示为0
		"""
		_current_state = np.zeros([9, 10, 9])
		_current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]

		if self.game_start:
			# 解析上一个动作move[self.last_move]
			move = move_id2move_action[self.last_move]
			start_position = int(move[0]), int(move[1])  # 起始位置[move]
			end_position = int(move[2]), int(move[3])  # 结束位置[move]

			# 计算落子位置，走子之前为-1，走子之后为1
			_current_state[7][start_position[0]][start_position[1]] = -1
			_current_state[7][end_position[0]][end_position[1]] = 1

		# 指出当前是哪个玩家走子[先手玩家为1，否则为0]
		if self.action_count % 2 == 0:
			_current_state[8][:, :] = 1.0

		return _current_state

	def do_move(self, move):
		"""
		根据move对棋盘装填做出改变
		"""
		self.game_start = True  # 游戏开始
		self.action_count += 1  # 移动次数加1
		move_action = move_id2move_action[move]  # 解析移动动作
		# 计算移动前后的坐标位置
		start_y, start_x = int(move_action[0]), int(move_action[1])
		end_y, end_x = int(move_action[2]), int(move_action[3])
		state_list = copy.deepcopy(self.state_deque[-1])
		# 判断是否吃子
		if state_list[end_y][end_x] != '一一':
			end_pos = state_list[end_y][end_x]
			# 如果吃掉对方的帅，则返回当前的current_player胜利
			if end_pos == '红帅':
				self.winner = self.color2id['黑']
			elif end_pos == '黑帅':
				self.winner = self.color2id['红']
			# 移动被吃掉的棋
			elif end_pos in self.have_pos:
				self.have_pos.remove(end_pos)
		else:
			self.kill_action += 1
		# 更改棋盘坐标状态
		state_list[end_y][end_x] = state_list[start_y][start_x]
		state_list[start_y][start_x] = '一一'
		# 改变当前玩家
		self.curr_player_color = '黑' if self.curr_player_color == '红' else '红'
		self.curr_player_id = 1 if self.curr_player_id == 2 else 2
		# 记录最后一次移动的位置
		self.last_move = move
		self.state_deque.append(np.array(state_list))

	def has_a_winner(self):
		"""
		是否产生赢家，一共有三种状态[红方胜，黑方胜，平局]
		"""
		if self.winner is not None:
			return True, self.winner
		# 平局先手判负
		elif self.kill_action >= CONFIG['kill_action'] or not self.have_pos:
			return False, -1
		return False, -1

	def game_end(self):
		"""
		判断当前对局是否结束
		"""
		# 计算赢、赢家
		win, winner = self.has_a_winner()
		if win:
			return True, winner
		# 平局，没有赢家
		elif self.kill_action >= CONFIG['kill_action']:
			return True, -1

		return False, -1
	
	def get_curr_player_color(self):
		"""
		获取当前玩家棋子花色[红/黑]
		"""
		return self.curr_player_color
	
	def get_curr_player_id(self):
		"""
		获取当前玩家棋子ID
		"""
		return self.curr_player_id