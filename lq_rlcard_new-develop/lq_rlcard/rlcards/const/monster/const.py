# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum

PLAYER_NUMS = 4  # 玩家数量
BASE_GOLD = 100.0  # 游戏底分

MONSTER = 10  # 师傅牌['T']
MAGIC_CARD = 20  # 万能牌['B']
PUPIL = [3, 8, 5]  # 徒弟牌['3', '8', '5']
BOGY = [11, 12, 13]  # 妖怪牌['J', 'Q', 'K']

INIT_REWARDS = [0.0, 0.0, 0.0, 0.0]  # 初始奖励
ALL_GOLDS = [1600.0, 1600.0, 1600.0, 1600.0]  # 所有金币
BUST_FLAGS = {"down": False, "right": False, "up": False, "left": False}  # 破产标志

class ActionType(str, Enum):
    """
    动作
    """
    PICK_CARDS = "PICK_CARDS"

CARD_TYPES = {
    1: 'D',  # 方块♦
    2: 'C',  # 梅花♣
    3: 'H',  # 红心♥
    4: 'S',  # 黑桃♠
    5: 'M'  # 万能牌
}

MAGIC = 520  # 万能牌

ALL_CARDS = (
    103, 105, 108, 110, 111, 112, 113,  # 方块
    203, 205, 208, 210, 211, 212, 213,  # 梅花
    303, 305, 308, 310, 311, 312, 313,  # 红心
    403, 405, 408, 410, 411, 412, 413,  # 黑桃
    )

ROUND_CARDS = [
    103, 105, 108, 110, 111, 112, 113,
    203, 205, 208, 210, 211, 212, 213,
    303, 305, 308, 310, 311, 312, 313,
    403, 405, 408, 410, 411, 412, 413,
    520, 520, 520, 520
]

# 卡牌类型(花色 -> [黑桃、红心、方块、梅花])
CARD_TYPE = ['S', 'H', 'D', 'C', 'BM']
CARD_TYPE_STR = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣', 's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}

# 卡牌(4 x 7 + 4 = 32)
CARD_VALUE_STR = ['J', 'Q', 'K', '3', '8', '5', 'T', 'B']  # 'T' = 10 点, 'B' = 万能牌
CARD_VALUE_STR_INDEX = {'J': 0, 'Q': 1, 'K': 2, '3': 3, '8': 4, '5': 5, 'T': 6, 'B': 7, 'PICK_CARDS': 8}

# 索引
CARD_RANK_OLD = ['J', 'Q', 'K', '3', '8', '5', 'T', 'BM']
CARD_RANK = [11, 12, 13, 3, 8, 5, 10, 20]

# Card2Column = {'J': 0, 'Q': 1, 'K': 2, '3': 3, '8': 4, '5': 5, 'T': 6, 'B': 7}
Card2Column = {11: 0, 12: 1, 13: 2, 3: 3, 8: 4, 5: 5, 10: 6, 20: 7, 'PICK_CARDS': 8}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

# 动作解析常量
action_dict = {
    11: 'J',
    12: 'Q',
    13: 'K',
    3: '3',
    8: '8',
    5: '5',
    10: 'T',
    20: 'B'
}

update_action_dict = {
    0: 11,
    1: 12,
    2: 13,
    3: 3,
    4: 8,
    5: 5,
    6: 10,
    7: 20,
    8: 99
}

CARD_VALUES = {
    11: -0.0015,
    12: -0.0025,
    13: -0.0035,
    3: -0.015,
    8: -0.025,
    5: -0.035,
    10: -0.07,
    20: -0.12
}

# 测试输出数据
if __name__ == '__main__':
    print('输出 CARD_TYPE: ', CARD_TYPE)
    print('输出 CARD_TYPE_STR: ', CARD_TYPE_STR)
    print('输出 CARD_VALUE_STR: ', CARD_VALUE_STR)
    print('输出 CARD_VALUE_STR_INDEX: ', CARD_VALUE_STR_INDEX)
    print('输出 Card2Column: ', Card2Column)
    print('输出 NumOnes2Array: ', NumOnes2Array)