from cmath import pi
import random
from rlcards.games.pig.kong_pai.poker_const import CombType, pokers_map, LianDuiNum
from rlcards.games.pig.kong_pai.gen_comb import GenComb
from copy import deepcopy
import numpy as np

POKERS_MAP = ...


def randint(a, b):
    """ [a, b] """
    return random.randint(a, b)


def random_choice_num(num_arr: list, rate_arr=None):
    """ 抽取随机数 """
    if not rate_arr:
        rate_arr = []
    if not num_arr:
        return
    if rate_arr:
        extra_num = len(num_arr) - len(rate_arr)
        extra_num > 0 and rate_arr.extend([0 for _ in range(extra_num)])
        if extra_num < 0:
            for _ in range(-1 * extra_num):
                rate_arr.pop()
    if sum(rate_arr) != 1:
        rate_arr = None
    return np.random.choice(a=num_arr, size=1, replace=True, p=rate_arr)[0]


def deal_cards_alg(cards: dict, yet_cards_len=0, lian_zha=False):
    """
    发牌算法
    cards: ps: -> {3: [102, 203, 303, 403], ...}
    """
    if not yet_cards_len:
        all_comb = CombType.all_value()  # 返回所有类型的列表
        rand_comb = random.sample(all_comb, 1)[0]  # 随机选一个类型

    else:
        all_comb = CombType.all_value()
        rand_comb = random.sample(all_comb, 1)[0]
    gen_map = {
        CombType.ZHA_DAN: GenComb.get_zha_dan,
        CombType.SHUN_ZI: GenComb.get_shun_zi,
        CombType.LIAN_DUI: GenComb.get_lian_dui,
        CombType.FEI_JI: GenComb.get_fei_ji,
        CombType.DUI_ZI: GenComb.get_pairs,
    }

    comb_count = {
        CombType.ZHA_DAN: 4,
        CombType.SHUN_ZI: 1,
        CombType.LIAN_DUI: 2,
        CombType.FEI_JI: 3,
        CombType.DUI_ZI: 2,
    }

    # num_map = {
    #     CombType.ZHA_DAN: LianZhaNum.MIN_NUM if not lian_zha else randint(LianZhaNum.MIN_NUM, LianZhaNum.MAX_NUM),
    #     CombType.SHUN_ZI: randint(ShunNum.MIN_NUM, ShunNum.MAX_NUM),
    #     CombType.LIAN_DUI: randint(LianDuiNum.MIN_NUM, LianDuiNum.MAX_NUM),
    #     CombType.FEI_JI: randint(FeiJiNum.MIN_NUM, FeiJiNum.MAX_NUM),
    #     CombType.DUI_ZI: 1,
    # }

    get_comb = gen_map.get(rand_comb)
    count = comb_count.get(rand_comb) or 0
    if rand_comb == CombType.LIAN_DUI:
        num = random_choice_num([3, 4, LianDuiNum.MAX_NUM], [0.8, 0.15, 0.05])
        res_cards = GenComb.get_lian_dui(cards, repeat_num_max=num)
        res_cards = res_cards and random.sample(res_cards, 1)
        return res_cards and res_cards[0] * count

    if get_comb and callable(get_comb):
        # comb_len = num_map.get(rand_comb) or 1
        res_cards = get_comb(cards) or []
        # print(res_cards)
        # print("rand_comb:", rand_comb, "count:", count)
        res_cards = res_cards and random.sample(res_cards, 1)
        return res_cards and res_cards[0] * count


def check_all_in_pile(POKERS_MAP, res):
    res_dict = {}
    flag = True
    for key in res:
        res_dict[key] = res_dict.get(key, 0) + 1

    for k, v in res_dict.items():
        if not k in POKERS_MAP:
            flag = False
        elif k in POKERS_MAP and len(POKERS_MAP.get(k)) < v:
            flag = False

    return flag


def cal_time(func):
    def inner(*args, **kwargs):
        import time
        s = time.time()
        func(*args, **kwargs)
        print(func.__name__, "耗时: ", time.time() - s)

    return inner


def gen_none_type_pile(list):
    '''
    生成没有花色的牌堆
    '''
    non_type_pile = []
    for card in list:
        if 2 < card % 100 < 15:
            non_type_pile.append(card % 100)
        elif card % 100 == 16:
            non_type_pile.append(card % 100 + 1)
        elif card % 100 == 18:
            non_type_pile.append(card % 100 + 2)
        elif card % 100 == 20:
            non_type_pile.append(card % 100 + 10)
        else:
            return ValueError
    return non_type_pile


def cal_count(list1, list2, list3, dict):
    length = 0
    for v in dict.values():
        if v:
            length += len(v)
    return len(list1) + len(list2) + len(list3) + length


# @cal_time
def three_player_cards():
    list1 = []
    list2 = []
    list3 = []
    global POKERS_MAP
    POKERS_MAP = deepcopy(pokers_map)
    for i in range(30):
        res = deal_cards_alg(POKERS_MAP)
        if check_all_in_pile(POKERS_MAP, res):
            templist = []
            if len(list1) + len(res) <= 17 or len(list2) + len(res) <= 17 or len(list3) + len(res) <= 17:
                while res:
                    temp = res.pop()
                    card = POKERS_MAP[temp][random.randint(0, len(POKERS_MAP[temp]) - 1)]
                    templist.append(card)
                    POKERS_MAP[temp].pop(POKERS_MAP[temp].index(card))

                if len(list1) + len(templist) <= 17:
                    while (templist):
                        list1.append(templist.pop())
                elif len(list2) + len(templist) <= 17:
                    while (templist):
                        list2.append(templist.pop())
                elif len(list3) + len(templist) <= 17:
                    while (templist):
                        list3.append(templist.pop())

    list1, list2, list3, list4 = append_cards_to_17(list1, list2, list3, dict_remain_cards())

    return list1, list2, list3, list4


def dict_remain_cards():
    templist = []
    for v in POKERS_MAP.values():
        if v:
            for element in v:
                templist.append(element)
    return templist


def append_cards_to_17(list1, list2, list3, list4):
    while len(list1) < 17 or len(list2) < 17 or len(list3) < 17:
        random.shuffle(list4)
        card = list4.pop()
        if len(list1) < 17:
            list1.append(card)
        elif len(list2) < 17:
            list2.append(card)
        elif len(list3) < 17:
            list3.append(card)

    return list1, list2, list3, list4


def ceshi():
    for i in range(1):
        list1, list2, list3, list4 = three_player_cards()
        print(list1)
        print(list2)
        print(list3)
        print(list4)
        print(gen_none_type_pile(list1))
        print(gen_none_type_pile(list2))
        print(gen_none_type_pile(list3))
        print(gen_none_type_pile(list4))

        print(len(list1) + len(list2) + len(list3) + len(list4))

        all_list = list1 + list2 + list3 + list4
        test_list = [103, 203, 303, 403] + [104, 204, 304, 404] + [105, 205, 305, 405] + [106, 206, 306, 406] + [107,
                                                                                                                 207,
                                                                                                                 307,
                                                                                                                 407] + [
                        108, 208, 308, 408] + [109, 209, 309, 409] + [110, 210, 310, 410] + [111, 211, 311, 411] + [112,
                                                                                                                    212,
                                                                                                                    312,
                                                                                                                    412] + [
                        113, 213, 313, 413] + [114, 214, 314, 414] + [116, 216, 316, 416] + [518, 520]

        all_dict = {}
        test_dict = {}
        for key in all_list:
            all_dict[key] = all_dict.get(key, 0) + 1
        for key in test_list:
            test_dict[key] = test_dict.get(key, 0) + 1

        print(all_dict)
        print(test_dict)

        for k, v in all_dict.items():
            if not (test_dict[k] and v == 1):
                raise ValueError(".....................")
        global POKERS_MAP
        POKERS_MAP = deepcopy(pokers_map)


if __name__ == '__main__':
    ceshi()
