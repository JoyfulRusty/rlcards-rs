# -*- coding: utf-8 -*-

from rlcards.games.aichess.alphabeta import const

from step import Step

def calc_legal_move(list_pieces, from_x, from_y, to_x, to_y, mg_init):
	"""
	计算合法移动位置
	玩家2表示机器人
	"""
	pieces = move_by_deep(list_pieces, from_x, from_y, to_x, to_y, mg_init)
	return [pieces[0].x, pieces[0].y, pieces[1], pieces[2]]

def move_by_deep(list_pieces, x1, y1, x2, y2, mg_init):
	"""
	移动深度
	"""
	old_step = Step(8 - x1, y1, 8 - x2, y2)
	mg_init.move_to(old_step)
	mg_init.alpha_by_beta(const.max_depth, const.min_val, const.max_val)
	new_step = mg_init.best_move
	mg_init.move_to(new_step)
	list_move_enable = []
	for i in range(0, 9):
		for j in range(0, 10):
			for item in list_pieces:
				if item.x == 8 - new_step.from_x and item.y == new_step.from_y:
					list_move_enable.append([item, 8 - new_step.to_x, new_step.to_y])
	pieces_best = list_move_enable[0]
	return pieces_best