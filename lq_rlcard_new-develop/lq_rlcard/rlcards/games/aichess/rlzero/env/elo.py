# -*- coding: utf-8 -*-

import math


# todo: 计算ELO象棋等级评分

def excepted_score(rating1, rating2):
	"""
	期望分数
	"""
	return 1 / (1 + math.pow(10, (rating1 - rating2) / 400))

def update_rating(rating, excepted_score, actual_score, k_factor):
	"""
	更新速率
	"""
	return rating + k_factor * (actual_score - excepted_score)

def elo_cal(player1_rating, player2_rating, player1_score, player2_score, k_factor=32):
	"""
	计算象棋elo分数
	"""
	excepted1_score = excepted_score(player1_rating, player2_rating)
	excepted2_score = excepted_score(player2_rating, player1_rating)
	new_player1_rating = update_rating(player1_rating, excepted1_score, player1_score, k_factor)
	new_player2_rating = update_rating(player2_rating, excepted2_score, player2_score, k_factor)

	return new_player1_rating, new_player2_rating