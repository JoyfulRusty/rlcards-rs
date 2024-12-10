# -*- coding: utf-8 -*-
import itertools

import torch
import numpy as np

from predict.ddz.xxc.const import Env2IdxMap, RealCard2EnvCard


def encode2onehot(cards):
	"""
	将牌转换为one-hot编码
	"""
	cards = [Env2IdxMap[i] for i in cards]
	onehot = torch.zeros((4, 15))
	for i in range(15):
		onehot[:cards.count(i), i] = 1
	return onehot

def encode2onehot_by_real(cards):
	"""
	将牌转换为one-hot编码，不使用映射
	"""
	cards = [RealCard2EnvCard[c] for c in cards]
	Onehot = torch.zeros((4, 15))
	for i in range(0, 15):
		Onehot[:cards.count(i), i] = 1
	return Onehot

def rank_arr(arr):
	"""
	得到arr的排名数组
	"""
	# 将数组排序并返回原始索引
	sort_idx = np.argsort(arr)
	# 排名数组
	rank = np.empty_like(sort_idx)
	rank[sort_idx] = np.arange(len(arr))
	# 将排名加1以得到1开始的排名
	return rank + 1

def select(cards, num):
	return [list(i) for i in itertools.combinations(cards, num)]