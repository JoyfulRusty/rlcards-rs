# -*- coding: utf-8 -*-

import json

from collections import OrderedDict
from rlcards.data.doudizhu.data_path import \
    CARD_TYPE_PATH, \
    ACTION_SPACE_PATH, \
    TYPE_CARD_PATH

def read_card_data(path):
    with open(path, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
        CARD_TYPE_DATA = (data, list(data), set(data))
    return CARD_TYPE_DATA


def read_action_data(path):
    with open(path, 'r') as f:
        ID_2_ACTION = f.readline().strip().split()
        ACTION_2_ID = {}
        for i, action in enumerate(ID_2_ACTION):
            ACTION_2_ID[action] = i
    return ACTION_2_ID, ID_2_ACTION

def read_type_data(path):
    with open(path, 'r') as f:
        TYPE_CARD_DATA = json.load(f, object_pairs_hook=OrderedDict)
    return TYPE_CARD_DATA

CARD_TYPE_DATA = read_card_data(CARD_TYPE_PATH)
ACTION_2_ID, ID_2_ACTION = read_action_data(ACTION_SPACE_PATH)
TYPE_CARD_DATA = read_type_data(TYPE_CARD_PATH)

# 测试数据输出
if __name__ == '__main__':
    # print("CARD_TYPE_DATA: ", CARD_TYPE_DATA)
    print("ACTION_2_ID: ", ACTION_2_ID)
    # print("ID_2_ACTION: ", ID_2_ACTION)
    # print("TYPE_CARD_DATA: ", TYPE_CARD_DATA)