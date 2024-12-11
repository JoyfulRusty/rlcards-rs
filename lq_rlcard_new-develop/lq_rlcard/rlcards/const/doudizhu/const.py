# -*- coding: utf-8 -*-

from collections import OrderedDict

PLAYER_NUMS = 3


CARD_SUIT = ['S', 'H', 'D', 'C', 'BJ', 'RJ']

CARD_RANK_STR = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K','A', '2', 'B', 'R']

CARD_RANK_STR_INDEX = {
    '3': 0,
    '4': 1,
    '5': 2,
    '6': 3,
    '7': 4,
    '8': 5,
    '9': 6,
    'T': 7,
    'J': 8,
    'Q': 9,
    'K': 10,
    'A': 11,
    '2': 12,
    'B': 13,
    'R': 14
    }

# 排行榜
CARD_RANK = ['3',
             '4',
             '5',
             '6',
             '7',
             '8',
             '9',
             'T',
             'J',
             'Q',
             'K',
             'A',
             '2',
             'BJ',
             'RJ']

INDEX = {
    '3': 0,
    '4': 1,
    '5': 2,
    '6': 3,
    '7': 4,
    '8': 5,
    '9': 6,
    'T': 7,
    'J': 8,
    'Q': 9,
    'K': 10,
    'A': 11,
    '2': 12,
    'B': 13,
    'R': 14
    }

INDEX = OrderedDict(sorted(INDEX.items(), key=lambda t: t[1]))
# 测试输出数据
if __name__ == '__main__':
    print(INDEX)