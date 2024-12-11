# -*- coding: utf-8 -*-

import numpy as np

# todo: 创建array压缩

num2array = dict(
	{
		1: np.array([1, 0, 0, 0, 0, 0, 0]),
		2: np.array([0, 1, 0, 0, 0, 0, 0]),
		3: np.array([0, 0, 1, 0, 0, 0, 0]),
		4: np.array([0, 0, 0, 1, 0, 0, 0]),
		5: np.array([0, 0, 0, 0, 1, 0, 0]),
		6: np.array([0, 0, 0, 0, 0, 1, 0]),
		7: np.array([0, 0, 0, 0, 0, 0, 1]),
		8: np.array([-1, 0, 0, 0, 0, 0, 0]),
		9: np.array([0, -1, 0, 0, 0, 0, 0]),
		10: np.array([0, 0, -1, 0, 0, 0, 0]),
		11: np.array([0, 0, 0, -1, 0, 0, 0]),
		12: np.array([0, 0, 0, 0, -1, 0, 0]),
		13: np.array([0, 0, 0, 0, 0, -1, 0]),
		14: np.array([0, 0, 0, 0, 0, 0, -1]),
		15: np.array([0, 0, 0, 0, 0, 0, 0])
	}
)


def array2num(array):
	"""
	将array转换为数字
	"""
	return list(filter(lambda string: (num2array[string] == array).all(), num2array))[0]

def state_list2state_num_array(state_list):
	"""
	压缩状态存储
	"""
	_state_array = np.zeros([10, 9, 7])
	for i in range(10):
		for j in range(9):
			_state_array[i][j] = num2array[state_list[i][j]]
	return _state_array

# (state, mct_s_prob, winner) ((9,10,9),2086,1) => ((9,90),(2,1043),1)
def zip_state_mct_s_prob(tuple):
	"""
	压缩状态蒙特卡洛搜索概率
	"""
	state, mct_s_prob, winner = tuple
	state = state.reshape((9, -1))
	mct_s_prob = mct_s_prob.reshape((2, -1))
	state = zip_array(state)
	mct_s_prob = zip_array(mct_s_prob)
	return state, mct_s_prob, winner

def recovery_state_mct_s_prob(tuple):
	"""
	恢复状态蒙特卡洛搜索概率
	"""
	state, mct_s_prob, winner = tuple
	state = recovery_array(state)
	mct_s_prob = recovery_array(mct_s_prob)
	state = state.reshape((9, 10, 9))
	mct_s_prob = mct_s_prob.reshape(2086)
	return state, mct_s_prob, winner

def zip_array(array, data=0.0):
	"""
	压缩成稀疏数组
	numpy新版本不加dtype='object'会报错
	"""
	zip_res = []
	zip_res.append([len(array), len(array[0])])
	for i in range(len(array)):
		for j in range(len(array[0])):
			if array[i][j] != data:
				zip_res.append([i, j, array[i][j]])
	return np.array(zip_res, dtype='object')

def recovery_array(array, data=0.):
	"""
	恢复数组
	"""
	recovery_res = []
	for i in range(array[0][0]):
		recovery_res.append([data for i in range(array[0][1])])
	for i in range(1, len(array)):
		recovery_res[array[i][0]][array[i][1]] = array[i][2]
	return np.array(recovery_res)
