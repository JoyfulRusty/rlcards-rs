# -*- coding: utf-8 -*-

import numpy as np

from enum import IntEnum, unique

from base.base_enum import BaseEnum


class GradeScore(IntEnum):
    """ 等级分 """
    ZHA_DAN = 15
    DA_WANG = 10
    XIAO_WANG = 8
    VAL_2 = 5
    SHUN_ZI = 5
    LIAN_DUI = 5
    FEI_JI = 4
    DAN_FEI_JI = 3
    PAIRS = 2
    DAN = 1

Env2IdxMap = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 16: 12, 18: 13, 20: 14}

RealCard2EnvCard = {
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
    '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
    'K': 10, 'A': 11, '2': 12, 'X': 13, 'D': 14
}

BASE_SCORE = {
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 3,
    14: 3,
    16: 3,
    18: 4,
    20: 4,
}


Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 16: 12}

NumOnes2Array = {
    0: np.array([0, 0, 0, 0]),
    1: np.array([1, 0, 0, 0]),
    2: np.array([1, 1, 0, 0]),
    3: np.array([1, 1, 1, 0]),
    4: np.array([1, 1, 1, 1])
}


ALL_POSITION = ['landlord_up', 'landlord', 'landlord_down']
ALL_POSITION_4_AN = ['landlord1', 'landlord2', 'landlord3', "landlord4"]


FK_3 = 31  # 编码为31
DI_ZHU_CARD = 3

# 游戏环境中用于打印日志使用
EnvCard2RealCard = {
    3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
    13: 'K', 14: 'A', 16: '2', 18: 'X', 20: 'D'
}

# 斗地主所有单炸
BOMBS = [
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
    [7, 7, 7, 7],
    [8, 8, 8, 8],
    [9, 9, 9, 9],
    [10, 10, 10, 10],
    [11, 11, 11, 11],
    [12, 12, 12, 12],
    [13, 13, 13, 13],
    [14, 14, 14, 14],
    [16, 16, 16, 16],
    [18, 20]
]

# 经典暗地主所有牌(牌值)
AllEnvCard = [
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
    12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 16, 16, 16, 16, 18, 20
]

# 4人暗地主所有牌(牌值)
AllEnvCardAn4 = [
    FK_3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
    12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 16, 16, 16, 16, 18, 20
]

# douzero 2|小王|大王 转换为子游戏的 2|小王|大王
EXTRA_CARD_MAP = {
    18: 20,  # 小鬼
    20: 30,  # 大鬼
}

# 子游戏的 2|小王|大王 转换为douzero 2|小王|大王
EXTRA_CARD_MAP_VK = {
    17: 16,  # 2
    20: 18,  # 小鬼
    30: 20,  # 大鬼
}

BID_THRESHOLDS = [0, 0.3, 0]
BID_THRESHOLDS_BY_WINRATE = {
    1: -0.2,
    2: 0,
    3: 0.3,
}
BID_THRESHOLDS_AN = [1, 2, 1]
# 暗地主叫地主阈值
BID_THRESHOLDS_BY_AN_WINRATE = {
    1: 1,
    2: 2,
    3: 3
}
# 在地主是抢来的情况下
# 叫地主，超级加倍，加倍，阈值
CHAN_THRESHOLDS = ((0.3, 0.15), (0.5, 0.15))
# 使用规则判断是否进行抢地主
FARMER_CHAN_THRESHOLDS = (6, 1.2)

USE_RULE_LANDLORD_REQUIREMENTS = False  # 使用规则判断能否抢地主

# global parameters
MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

# action types
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
TYPE_5_KING_BOMB = 5
TYPE_6_3_1 = 6
TYPE_7_3_2 = 7
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11
TYPE_12_SERIAL_3_2 = 12
TYPE_13_4_2 = 13  # 四带2
TYPE_14_4_22 = 14  # 四带2对
TYPE_15_SERIAL_BOMB = 15  # 滚炸
TYPE_16_WRONG = 16  # 错误

# betting round action
PASS = 0
CALL = 1
RAISE = 2

S_KING = 18
B_KING = 20
CARD2POINT = 16



GHOST_CARDS = [518, 520]


@unique
class ActionType(BaseEnum):
    """ 动作类型 """
    TYPE_0_PASS = 0, "PASS", "0"
    TYPE_1_SINGLE = 1, "单张", "1"
    TYPE_2_PAIR = 2, "一对", "2"
    TYPE_3_TRIPLE = 3, "3不带", "3"
    TYPE_4_BOMB = 4, "4张炸弹", "4"
    TYPE_5_KING_BOMB = 5, "王炸", "5"
    TYPE_6_3_1 = 6, "三带一", "6"
    TYPE_7_3_2 = 7, "三带二", "7"
    TYPE_8_SERIAL_SINGLE = 8, "单连牌：最少5张", "8"
    TYPE_9_SERIAL_PAIR = 9, "连对", "9"
    TYPE_10_SERIAL_TRIPLE = 10, "飞机不带", "10"
    TYPE_11_SERIAL_3_1 = 11, "飞机带单", "11"
    TYPE_12_SERIAL_3_2 = 12, "飞机带双", "12"
    TYPE_13_4_2 = 13, "四带2", "13"
    TYPE_14_4_22 = 14, "四带2对", "14"
    TYPE_15_SERIAL_BOMB = 15, "滚炸", "15"
    TYPE_16_WRONG = 16, "错误", "16"
