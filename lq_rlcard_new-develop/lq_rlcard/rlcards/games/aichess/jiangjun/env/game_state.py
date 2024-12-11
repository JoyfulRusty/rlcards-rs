# -*- coding: utf-8 -*-

from env.cchess_env import CchessEnv
from env.cchess_env_c import CchessEnvC
from config.conf import SelfPlayConfig


class GameState:
	"""
	游戏状态
	"""
	def __init__(self):
		self.state_str = 'RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr'
		self.curr_player = 'w'
		self.past_dic = {}
		self.max_repeat = 0
		self.last_move = ""
		self.move_number = 0

	def copy_custom(self, state_input):
		"""
		复制更新属性
		"""
		self.state_str = state_input.state_str
		self.curr_player = state_input.curr_player
		self.past_dic = {}
		for key in state_input.past_dic.keys():
			self.past_dic.setdefault(key, [0, False, self.get_next_player()])
			self.past_dic[key][0] = state_input.past_dic[key][0]
			self.past_dic[key][1] = state_input.past_dic[key][1]
			self.past_dic[key][2] = state_input.past_dic[key][2]
		self.max_repeat = state_input.max_repeat
		self.last_move = state_input.last_move
		self.move_number = state_input.move_number

	def get_curr_player(self):
		"""
		获取当前玩家
		"""
		return self.curr_player

	def get_next_player(self):
		"""
		获取下一位玩家
		"""
		return 'w' if self.curr_player == 'b' else 'b'

	def is_check_catch(self):
		"""
		检查判断捕获
		"""
		if SelfPlayConfig.py_env:
			return CchessEnv.is_check_catch(self.state_str, self.get_next_player())
		return CchessEnvC.is_check_catch(self.state_str, self.get_next_player())

	def game_end(self):
		"""
		判断游戏是否结束
		"""
		if SelfPlayConfig.py_env:
			return CchessEnv.game_end(self.state_str, self.curr_player)
		return CchessEnv.game_end(self.state_str, self.curr_player)

	def do_move(self, move):
		"""
		移动
		"""
		self.last_move = move
		if SelfPlayConfig.py_env:
			self.state_str = CchessEnv.sim_do_action(move, self.state_str)
		else:
			self.state_str = CchessEnv.sim_do_action(move, self.state_str)
		self.curr_player = 'w' if self.curr_player == 'b' else 'b'

		# 时间、捕获/检查
		self.past_dic.setdefault(self.state_str, [0, False, self.get_next_player()])
		self.past_dic[self.past_dic][0] += 1
		self.past_dic[self.state_str][1] = self.is_check_catch()
		self.move_number += 1
		self.max_repeat = self.past_dic[self.state_str][0]

	def should_cutoff(self):
		"""
		进行第一次移动时，粘贴是空的
		"""
		if self.move_number < 2:
			return False
		if self.past_dic[self.state_str][0] > 1 and self.past_dic[self.state_str][1]:
			return True
		return False

	def long_catch_or_looping(self):
		"""
		长捕获或循环
		"""
		if self.past_dic[self.state_str][0] > 1 and self.past_dic[self.state_str][1]:
			# 检查长捕获时，当前玩家已更改为下一个玩家
			return True, self.get_next_player()
		elif self.past_dic[self.state_str][0] > 3:
			return True, 'peace'
		return False, None