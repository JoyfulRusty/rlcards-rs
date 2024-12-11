# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum, IntEnum, unique

# TODO: 设置常量
PLAYER_NUMS = 4

ALL_CARDS = (
	11, 12, 13, 14, 15, 16, 17, 18, 19,  # 万
	21, 22, 23, 24, 25, 26, 27, 28, 29,  # 条
	31, 32, 33, 34, 35, 36, 37, 38, 39,  # 筒
)

CARD_TYPES = {
	1: "chars",  # 万
	2: "bamboos",  # 条
	3: "dots",  # 筒
}

CARD_VALUES = [
	'11', '12', '13', '14', '15', '16', '17', '18', '19',
	'21', '22', '23', '24', '25', '26', '27', '28', '29',
	'31', '32', '33', '34', '35', '36', '37', '38', '39',
]

Card2StrIdx = {
	0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18, 8: 19,
	9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29,
	18: 31, 19: 32, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39
}

Card2Column = {
	11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8,
	21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17,
	31: 18, 32: 19, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26,
	'GUO': 27, 'PONG': 28, 'MING_GONG': 29, 'SUO_GONG': 30, 'AN_GONG': 31, 'BAO_TING': 32, 'KAI_HU': 33
}

# 卡牌类型编码信息(万、条、筒、癞子)
CardType2Column = {
	1: 0,  # 万
	2: 1,  # 条
	3: 2,  # 筒
}

NumOnes2Array = {
	0: np.array([0, 0, 0, 0]),  # 0次
	1: np.array([1, 0, 0, 0]),  # 1次
	2: np.array([1, 1, 0, 0]),  # 2次
	3: np.array([1, 1, 1, 0]),  # 3次
	4: np.array([1, 1, 1, 1])   # 4次
}


@unique
class GameOverType(IntEnum):
	"""
	游戏结束类型
	"""
	DEFAULT = 0  # 默认值
	LIU_JU = 1  # 流局
	KAI_HU = 2  # 胡开


@unique
class SuitType(IntEnum):
	"""
	花色类型
	"""
	SUIT_C = 1  # 万 chars
	SUIT_B = 2  # 条 bamboos
	SUIT_D = 3  # 筒 dots


@unique
class CardType(IntEnum):
	"""
	卡牌类型
	"""
	YAO_JI = 21
	WU_GU_JI = 38
	LAI_ZI = 51

@unique
class CombType(IntEnum):
	"""
	连牌类型
	"""
	TYPE_S = 1  # 单张
	TYPE_KZ = 2  # 刻子
	TYPE_SZ = 3  # 顺子
	TYPE_SG = 4  # 四杠
	TYPE_DZ = 5  # 对子
	TYPE_S2Z = 6  # 顺2
	TYPE_JG = 7  # 间隔顺


@unique
class ActionType(str, Enum):
	"""
	游戏动作类型
	"""
	GUO = "GUO"  # 过
	PONG = "PONG"  # 碰
	KAI_HU = "KAI_HU"  # 胡
	AN_GONG = "AN_GONG"  # 暗杠
	SUO_GONG = "SUO_GONG"  # 索杠
	BAO_TING = "BAO_TING"  # 报听
	MING_GONG = "MING_GONG"  # 明杠

# 动作优先级别判断
ACTION_PRIORITY = {
	ActionType.GUO: 0,
	ActionType.PONG: 1,
	ActionType.MING_GONG: 2,
	ActionType.SUO_GONG: 3,
	ActionType.AN_GONG: 4,
	ActionType.BAO_TING: 5,
	ActionType.KAI_HU: 6
}


@unique
class HuType(IntEnum):
	"""
	胡牌类型
	"""
	PING_HU = 101  # 平胡
	DA_DUI_ZI = 102  # 大对子
	QI_DUI = 103  # 七对
	LONG_QI_DUI = 104  # 龙七对
	QING_YI_SE = 105  # 清一色
	QING_DA_DUI = 106  # 清大对
	QING_QI_DUI = 107  # 清七对
	QING_LONG_BEI = 108  # 清龙背
	DI_LONG_QI_DUI = 109  # 地龙七对
	SDI_LONG_QI = 110  # 双地龙七
	QING_DI_LONG = 111  # 清地龙
	QING_SDI_LONG = 112  # 清双地龙
	SAN_DI_LONG_QI = 113  # 三地龙七
	QING_SAN_DI_LONG = 114  # 清三地龙


# 胡牌类型分数映射
HT_SCORE_MAP = {
	HuType.PING_HU: 1,
	HuType.DA_DUI_ZI: 5,
	HuType.QI_DUI: 10,
	HuType.LONG_QI_DUI: 20,
	HuType.QING_YI_SE: 10,
	HuType.QING_DA_DUI: 15,
	HuType.QING_QI_DUI: 20,
	HuType.QING_LONG_BEI: 30,
	HuType.DI_LONG_QI_DUI: 20,
	HuType.SDI_LONG_QI: 30,
	HuType.QING_DI_LONG: 30,
	HuType.QING_SDI_LONG: 40,
	HuType.SAN_DI_LONG_QI: 40,
	HuType.QING_SAN_DI_LONG: 50
}


@unique
class ExtraHuType(IntEnum):
	"""
	额外胡牌
	"""
	TIAN_HU = 201  # 天胡
	DI_HU = 202  # 地胡
	TIAN_TING = 203  # 天听
	SHA_BAO = 204  # 杀报
	GANG_SHANG_HUA = 205  # 杠上花
	QIANG_GANG_HU = 206  # 抢杠胡
	GANG_SHANG_PAO = 207  # 杠上炮(热炮)
	BAO_TING_QING_QI_DUI = 208  # 报听清七对
	BAO_TING_QING_LONG_QI_DUI = 209  # 报听清龙七对

# 额外胡牌类型分数映射
EHT_SCORE_MAP = {
	ActionType.MING_GONG: 3,
	ActionType.SUO_GONG: 3,
	ActionType.AN_GONG: 3,

	ExtraHuType.DI_HU: 10,
	ExtraHuType.TIAN_HU: 10,
	ExtraHuType.SHA_BAO: 10,
	ExtraHuType.TIAN_TING: 10,
	ExtraHuType.QIANG_GANG_HU: 0,
	ExtraHuType.GANG_SHANG_PAO: 3,
	ExtraHuType.GANG_SHANG_HUA: 3,
}


class JiType(IntEnum):
	"""
	鸡/杠类型
	"""
	DEFAULT = 201  # 默认鸡
	CF_JI = 202  # 冲锋鸡
	ZR_JI = 203  # 责任鸡
	MT_JI = 204  # 满堂鸡
	SX_JI = 205  # 上下鸡
	WG_JI = 206  # 乌骨鸡
	WG_CFJ = 207  # 乌骨冲锋鸡
	WG_ZRJ = 208  # 乌骨责任鸡
	JIN_JI = 209  # 金鸡
	WG_JIN_JI = 210  # 乌骨金鸡
	CF_JIN_JI = 211  # 冲锋金鸡
	WG_CF_JIN_JI = 212  # 乌骨冲锋金鸡

	BEN_JI = 213  # 本鸡
	YIN_JI = 214  # 银鸡

	MING_GONG = 301  # 明杠
	SUO_GONG = 302  # 梭杠
	AN_GONG = 302  # 暗杠

JiType_SCORE_MAP = {
	# 鸡牌分(默认鸡)
	CardType.YAO_JI: 1,
	CardType.WU_GU_JI: 2,

	# 鸡牌类型分
	JiType.DEFAULT: 1,
	JiType.CF_JI: 3,
	JiType.CF_JIN_JI: 9,
	JiType.ZR_JI: 2,
	JiType.JIN_JI: 2,
	JiType.WG_JI: 2,
	JiType.WG_CFJ: 4,
	JiType.WG_ZRJ: 2,
	JiType.WG_JIN_JI: 4,
	JiType.WG_CF_JIN_JI: 12,
	JiType.YIN_JI: 2,
	JiType.AN_GONG: 3,
	JiType.SUO_GONG: 3,
	JiType.MING_GONG: 3
}