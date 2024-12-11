# -*- coding: utf-8 -*-

"""
todo: 中国象棋
	1.获取当前可行动作
	2.step(更新)
	3.判断游戏是否结束
	4.编码state_str为plane
"""

import os
import copy
import numpy as np

from ctypes import *
from enum import IntEnum


FULL_INIT_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
y_axis = '9876543210'
x_axis = 'abcdefghi'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
library = cdll.LoadLibrary(os.path.join(DIR_PATH, "libjiangjun32.so"))


class CchessEnvC:
	"""
	中国象棋环境
	"""
	def __init__(self):
		self.name = "a chess env"

	@staticmethod
	def sim_do_action(action, state_str):
		"""
		执行动作
		"""
		library.sim_do_action.argtypes = [c_char_p, c_char_p, c_char_p]
		library.sim_do_action.restype = c_void_p

		input_action = create_string_buffer(action.encode('utf-8'))
		input_state = create_string_buffer(state_str.encode('utf-8'))

		res_pt = (c_char * 150)()
		library.sim_do_action(input_action, input_state, res_pt)
		return res_pt.value.decode('utf-8')

	@staticmethod
	def is_check_catch(state_str, next_player):
		"""
		检查捕获
		"""
		library.is_check_catch.argtypes = [c_char_p, c_char_p]
		library.is_check_catch.restype = c_int8

		input_player = create_string_buffer(next_player.encode('utf-8'))
		input_state = create_string_buffer(state_str.encode('utf-8'))

		check_catch = library.is_check_catch(input_state, input_player)

		return check_catch

	@staticmethod
	def game_end(state_str, player):
		"""
		判断游戏是否结束
		"""
		library.game_end.argtypes = [c_char_p, c_char_p, c_char_p]
		library.game_end.restype = c_int8

		input_player = create_string_buffer(player.encode('utf-8'))
		input_state = create_string_buffer(state_str.encode('utf-8'))

		res_pt = (c_char * 5)()
		over = library.game_end(input_state, input_player, res_pt)
		winner = res_pt.value.decode('utf-8')

		return over, winner

	@staticmethod
	def get_legal_action(state_str, player):
		"""
		获取合法动作
		"""
		library.get_legal_action.argtypes = [c_char_p, c_char_p, c_char_p]
		library.get_legal_action.restype = c_void_p

		input_player = create_string_buffer(player.encode('utf-8'))
		input_state = create_string_buffer(state_str.encode('utf-8'))

		res_pt = (c_char * 1500)()
		library.get_legal_action(input_state, input_player, res_pt)
		legal_actions = res_pt.value.decode('utf-8')
		legal_actions = legal_actions[1:-1].split('/')

		return legal_actions