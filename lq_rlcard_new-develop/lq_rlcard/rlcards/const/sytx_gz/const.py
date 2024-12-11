# -*- coding: utf-8 -*-

# todo：水鱼天下游戏常量

import numpy as np

from enum import IntEnum, unique

PLAYER_NUMS = 2
GAME_OVER = 2

# ['S', 'H', 'C', 'D']  # 黑桃、红桃、梅花、方块
ALL_CARDS = [
	101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,  # 方块
	201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213,  # 梅花
	301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313,  # 红心
	401, 402, 403, 404, 405, 406, 407, 408, 409, 411, 412, 413,  # 黑桃
]

CARD_INDEX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]


CARD_TYPES = {
	1: "D",  # 方块
	2: "C",  # 梅花
	3: "H",  # 红桃
	4: "S",  # 黑桃
}

NumOnes2Array = {
	0: np.array([0, 0, 0, 0]),  # 0次
	1: np.array([1, 0, 0, 0]),  # 1次
	2: np.array([1, 1, 0, 0]),  # 2次
	3: np.array([1, 1, 1, 0]),  # 3次
	4: np.array([1, 1, 1, 1])   # 4次
}


# todo: 比牌
COMPARE_Z_SHA_X_FAN_NUM = 1  # 庄杀闲反
COMPARE_Z_SHA_X_XIN_NUM = -1  # 庄杀闲信
COMPARE_Z_ZOU_X_FAN_NUM = 2  # 庄走闲反
COMPARE_Z_ZOU_X_XIN_NUM = 0  # 庄走闲信
COMPARE_X_MI_Z_KAI_NUM = 2  # 闲密庄开
COMPARE_X_MI_Z_XIN_NUM = -2  # 闲密庄信
COMPARE_X_QG_Z_KAI_NUM = 2  # 闲强攻庄必开
COMPARE_Z_AND_X_SY = 3  # 庄、闲水鱼
COMPARE_X_SY = 4  # 闲水鱼
COMPARE_Z_SY = 5

COMPARE_REAL_WINNER = 2

GT_TWO = 1  # 大两铺
LT_ONE_B = 2  # 小于大铺
LT_ONE_S = 3  # 小于小铺
LT_TWO = 4  # 两铺都小

IS_EQUAL = 1  # 相等
IS_EQUAL_INT = 2
IS_MORE = 3  # 大于
IS_LESS = 4  # 小于
IS_DRAW = 5  #
IS_ILLEGAL = 6  # 非法
IS_MORE_PAI = 7  # 大于

SA_PU_QIANG_GONG = 1  # 撒扑强攻
SA_PU_MI = 2  # 撒扑密
SA_PU_LP = 3  #
SA_PU_MP = 4  #
SA_PU_SY = 5  # 撒扑水鱼
SA_PU_SY_TX = 6  # 撒扑水鱼天下
SA_PU_SY_MP = 7

DEAL_WIN = 1
PLAYER_WIN = 2
DRAW = 3


@unique
class ActionType(IntEnum):
	"""
	水鱼天下动作
	"""
	SP = 900  # 撒扑
	QG = 901  # 强攻
	MI = 902  # 密牌
	ZOU = 903  # 走
	SHA = 904  # 杀
	REN = 905  # 认
	XIN = 906  # 信
	FAN = 907  # 反
	KAI = 908  # 开


@unique
class LandLordAction(IntEnum):
	"""
	庄家动作
	"""
	ZOU = 903  # 走
	SHA = 904  # 杀
	REN = 905  # 认
	KAI = 908  # 开


@unique
class FarmerAction(IntEnum):
	"""
	闲家动作
	"""
	SP = 900  # 撒扑
	QG = 901  # 强攻
	MI = 902  # 密牌
	XIN = 906  # 信
	FAN = 907  # 反


@unique
class LzType(IntEnum):
	"""
	癞子类型(梦幻、广西)
	"""
	DEFAULT = 0
	MH = 1 # 梦幻
	GX = 2 # 广西


@unique
class SyType(IntEnum):
	"""
	水鱼类型
	"""
	SINGLE = 1  # 单张
	PAIR = 2  # 一对
	SY = 3  # 水鱼
	SY_TX = 4  # 水鱼天下


if __name__ == "__main__":
	print("===== 输出水鱼动作类型 =====")
	action_type = [print("输出水鱼动作类型: ", _) for _ in ActionType]
	print()

	print("===== 输出水鱼庄家动作类型 =====")
	landlord_action = [print("输出水鱼庄家动作类型: ", _) for _ in LandLordAction]
	print()

	print("===== 输出水鱼闲家动作类型 =====")
	farmer_action = [print("输出水鱼闲家动作类型: ", _) for _ in FarmerAction]