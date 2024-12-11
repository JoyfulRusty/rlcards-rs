# -*- coding: utf-8 -*-

class Game(object):
	"""
	表示2人棋盘游戏的游戏板及其逻辑
	"""
	def __init__(self):
		"""
		使用初始板状态初始化游戏
		"""
		pass

	def clone(self):
		"""
		todo: 创建游戏对象的深层克隆
		"""
		pass

	def play_action(self, action):
		"""
		todo: 在游戏界面进行动作
		动作:(行、列)形式的元组
		"""
		pass

	def get_valid_moves(self, current_player):
		"""
		todo: 返回移动及其有效性的列表

		在电路板中搜索零(0)。0表示一个空正方形
		包含以下形式的移动的列表(有效性、行、列)
		"""
		pass

	def check_game_over(self, current_player):
		"""
		todo: 检查游戏是否结束并返回可能的获胜者

		有3种可能的情况:
			1.比赛结束了，有一个赢家
			2.比赛结束了，但这是平局
			3.游戏还没有结束

		[winner: 1, loser: -1, draw: 0]
		"""
		pass