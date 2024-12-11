# -*- coding: utf-8 -*-

import random
import itertools

from enum import IntEnum
from copy import deepcopy
from collections import defaultdict

from rlcards.games.mahjong.utils import random_choice_num
from rlcards.const.mahjong.const import CardType, HuPaiType, ACTION_TYPE_MING_GANG, ACTION_TYPE_ZHUAN_WAN_GANG

LOG_PRINT = False


class TypeScore(IntEnum):
    """
    TODO: 卡牌类型分数
    """
    EDGE_CARD = 19  # 边张
    GAP_CARD = 20  # 坎张
    PAIR = 35  # 对子
    XIAO_SHUN = 40  # 二顺
    SHUN_ZI = 100  # 三顺
    KE_ZI = 115  # 刻子


class Card2Type(IntEnum):
    """
    TODO: 拆牌之后的最小组合
    """
    SINGLE_ONE = 0  # 单张
    EDGE_CARD = 1  # 边张
    GAP_CARD = 2  # 坎张
    PAIR = 3  # 对子
    XIAO_SHUN = 4  # 两面(小顺)
    SHUN_ZI = 5  # 顺子
    KE_ZI = 6  # 刻子


# TODO: 牌组合能减少向听数的多少
# 组合能减少的向听数map,卡张|一对|小顺 减少1向听，顺子|刻子减少 2向听
# 不同组合卡牌，能减去有效向听的数量
COMB_REMOVE_XTS = {
    Card2Type.GAP_CARD: 1,  # 减去向听1
    Card2Type.EDGE_CARD: 1,  # 减去向听1
    Card2Type.PAIR: 1,  # 减去向听1
    Card2Type.XIAO_SHUN: 1,  # 减去向听1
    Card2Type.SHUN_ZI: 2,  # 减去向听(三连顺)2
    Card2Type.KE_ZI: 2,  # 减去向听(刻子)2
}


class MoveGenerator:
    """
    根据拆牌后，胡牌类型计算牌面 估值 做出决策
    """
    __slots__ = (
        "__hand_cards",
        "__hand_cards_len",
        "__cards_to_count",
        "__pong_gang_cards",
        "__curr_card",
        "qys_flag_len",
        "xqd_flag_len",
        "ddz_flag_len",
        "modify_flag",
        "max_hu_type",
        "piles",
        "left_count",
        "most_hand_cards_len",
        "res_cards_to_count",
        "others_hand_cards",
        "the_worst_xts_by_hu_type",
        "round_over_the_worst_xts_by_hu_type"
    )

    def __init__(self):
        """
        初始化手牌参数
        """
        self.__hand_cards = []  # 当前手牌
        self.__hand_cards_len = len(self.__hand_cards)  # 手牌数量
        self.__cards_to_count = {}  # 卡牌数量统计
        self.__pong_gang_cards = []  # 碰杠牌
        self.__curr_card = 0
        self.piles = []  # 碰杠操作
        self.left_count = 0  # 卡牌数量
        self.qys_flag_len = 10  # 清一色标识(手牌同一种花色卡牌数量大于9)
        self.xqd_flag_len = 4  # 小七对标识(当前手牌对子数量大于4)
        self.ddz_flag_len = 3  # 大队子标识(当前手牌对子数大于3)
        self.modify_flag = 7  # 做大牌条件限制
        self.most_hand_cards_len = 14  # 最大手牌数量
        self.res_cards_to_count = {}  # 剩余手牌数量
        self.others_hand_cards = []  # 其他玩家当前手牌与碰牌[[hc], [piles]]
        self.the_worst_xts_by_hu_type = ...  # 胡牌类型最坏的向听数
        self.round_over_the_worst_xts_by_hu_type = ...  # 胡牌类型最坏的向听数
        self.max_hu_type = ["lqd", "dlq", "ddz", "xqd"]

    def calc_optimal_play_cards(self):
        """
        interface
        TODO: 计算最优出牌(向听数最小情况下)
        """
        return self.calc_xts_by_normal_hu_type_old()

    def calc_can_pong(self, card):
        """
        interface
        计算是否能碰(该碰)，计算碰前碰后的向听数比对
        """
        return self.after_pong_xts_less(card)

    def calc_can_xqd_pong(self, card):
        """
        处理小七对是否碰
        """
        # 是否满足小七对碰操作条件
        if self.count_xqd_nums():
            print("开始做小七对牌型，当前对子数为: {}".format(self.count_xqd_nums()))
            return False
        return self.after_pong_xts_less(card)

    def calc_can_gang(self, card, gang_type):
        """
        interface
        计算是否杠：暗杠|明杠|转弯杠
        """
        return self.after_gang_xts_less(card, gang_type)

    def after_gang_xts_less(self, card, gang_type) -> bool:
        """
        判断碰杠以后向听数是否减少
        """
        cards_to_count = self.calc_cards_to_count()
        args = self.calc_cards_list_by_count()
        if gang_type != ACTION_TYPE_MING_GANG and (not card or isinstance(card, list)):
            can_gang = []
            if isinstance(card, list):
                can_gang = card
            else:
                for card, count in cards_to_count.items():
                    if count > 3:
                        can_gang.append(card)
            if not can_gang:
                return False
            card_num = 4
            xts_ = self.the_worst_xts_by_hu_type[HuPaiType.PING_HU]
            for card in can_gang:
                self.the_worst_xts_by_hu_type[HuPaiType.PING_HU] = xts_
                args_tup = deepcopy(args)
                forward_xts, _ = self.cal_xts_ping_hu(*args_tup)
                hand_cards_copy = self.__hand_cards[:]
                for _ in range(card_num):
                    hand_cards_copy.remove(card)
                one_list, two_list, three_list, four_list = args_tup
                if card_num == 4:
                    four_list.remove(card)

                self.__hand_cards = hand_cards_copy
                self.the_worst_xts_by_hu_type[HuPaiType.PING_HU] -= 2
                need_mz = len(hand_cards_copy) // 3
                after_xts, _ = self.cal_xts_ping_hu(one_list, two_list, three_list, four_list, need_mz=need_mz)
                print("输出杠牌之前向听数为: {}, 杠牌之后向听数为: {}".format(forward_xts, after_xts))
                if -2 < after_xts <= forward_xts:
                    return card
                return False
        else:
            card_num = 3
            # 明杠和转弯杠数量特殊处理
            if gang_type != ACTION_TYPE_MING_GANG:
                card_num = 4
                if gang_type == ACTION_TYPE_ZHUAN_WAN_GANG:
                    card_num = 1  # 转弯杠手牌仅为1张
            if cards_to_count.get(card, 0) < card_num:
                return False
            forward_xts, _ = self.cal_xts_ping_hu(*args)
            hand_cards_copy = self.__hand_cards[:]
            for _ in range(card_num):
                hand_cards_copy.remove(card)

            one_list, two_list, three_list, four_list = args
            if card_num == 1:
                one_list.remove(card)
            if card_num == 3:
                three_list.remove(card)
            if card_num == 4:
                four_list.remove(card)

            self.__hand_cards = hand_cards_copy
            if card_num != 1:
                self.the_worst_xts_by_hu_type[HuPaiType.PING_HU] -= 2
            need_mz = len(hand_cards_copy) // 3
            after_xts, _ = self.cal_xts_ping_hu(one_list, two_list, three_list, four_list, need_mz=need_mz)
            print("输出杠牌之前向听数为: {}, 杠牌之后向听数为: {}".format(forward_xts, after_xts))
            if -2 < after_xts <= forward_xts:
                return card
            return False

    def after_pong_xts_less(self, card) -> bool:
        """
        判断碰杠以后向听数是否减少
        """
        card_num = 2
        cards_to_count = self.calc_cards_to_count()
        if cards_to_count.get(card, 0) < card_num:
            return False
        args = self.calc_cards_list_by_count()
        forward_xts, _ = self.cal_xts_ping_hu(*args)
        hand_cards_copy = self.__hand_cards[:]
        for _ in range(card_num):
            hand_cards_copy.remove(card)

        one_list, two_list, three_list, four_list = args
        if card_num == 2:
            two_list.remove(card)

        self.__hand_cards = hand_cards_copy
        self.the_worst_xts_by_hu_type[HuPaiType.PING_HU] -= 2
        need_mz = len(hand_cards_copy) // 3
        after_xts, _ = self.cal_xts_ping_hu(one_list, two_list, three_list, four_list, need_mz=need_mz)
        print("输出碰牌之前向听数为: {}, 碰牌之后向听数为: {}".format(forward_xts, after_xts))
        if after_xts == -1:
            print("当前手牌碰牌后形成胡牌~~")
        if -2 < after_xts <= forward_xts:
            return True
        return False

    @staticmethod
    def judge_cards_type(cards: list):
        """
        判断卡牌组合类型
        """
        cards_len = len(cards)

        # TODO: 单张(长度为1)
        if cards_len == 1:
            return Card2Type.SINGLE_ONE

        # TODO: 两张(长度为2)
        if cards_len == 2:
            # 对子
            if cards[0] == cards[1]:
                return Card2Type.PAIR
            # 边张
            elif cards[0] + 1 == cards[1]:
                if cards[0] == 1 or cards[0] == 8:
                    return Card2Type.EDGE_CARD
                # 两张，顺子
                return Card2Type.XIAO_SHUN
            # 坎张(间隔顺子)
            elif cards[0] + 2 == cards[1]:
                return Card2Type.GAP_CARD

        # TODO: 顺子，刻子(长度为3)
        elif cards_len == 3:
            # 三连顺
            if cards[0] == cards[1] == cards[2]:
                return Card2Type.SHUN_ZI
            # 刻子(三同)
            elif cards[0] + 1 == cards[1] and cards[1] + 1 == cards[2]:
                return Card2Type.KE_ZI

    def update_attr(
            self,
            hand_cards,
            piles=None,
            left_count=0,
            others_hand_cards=None,
            remain_cards=None,
            curr_card=None):
        """
        TODO: 更新属性
        根据手牌数量，判断玩家胡牌类型
        计算不同类型卡牌最差向听数
        """
        self.piles = piles or []
        self.__hand_cards = hand_cards  # 当前手牌
        self.left_count = left_count or 0  # 卡盒中卡牌数量
        self.__curr_card = curr_card or 0  # 当前出牌
        self.__hand_cards_len = len(self.__hand_cards)  # 手牌数量
        self.others_hand_cards = others_hand_cards or []  # 其他玩家手牌
        self.res_cards_to_count = remain_cards or {suit * 10 + num: 4 for suit in range(1, 4) for num in range(1, 10)}

        # TODO: 碰杠数量(计算碰杠数)
        # 碰杠数 = 最多4个面子 - 手牌数 整除 3
        pong_gang_num = 4 - self.__hand_cards_len // 3

        # TODO: 胡牌类型 -> 平胡，大队子
        self.the_worst_xts_by_hu_type = {
            HuPaiType.PING_HU: (self.most_hand_cards_len - 5 - 1) - pong_gang_num * 2,
            HuPaiType.DA_DUI_ZI: (self.most_hand_cards_len - 5 - 1) - pong_gang_num * 2,
        }

        # TODO: 胡牌类型 -> 七对，龙七对
        # 手牌十四张时
        if self.__hand_cards_len == self.most_hand_cards_len:
            # 最坏还需6个对子叫牌
            self.the_worst_xts_by_hu_type[HuPaiType.QI_DUI] = (self.most_hand_cards_len // 2) - 1
            self.the_worst_xts_by_hu_type[HuPaiType.LONG_QI_DUI] = (self.most_hand_cards_len // 2) - 1

        # TODO: 地龙七
        if self.__hand_cards_len == 11:
            # 地龙七已经有一个碰（最坏还需5个对子叫牌）
            self.the_worst_xts_by_hu_type[HuPaiType.DI_LONG_QI] = (self.most_hand_cards_len // 2) - 2

    def get_round_over_by_update_attr(self, hand_cards):
        """
        计算真人玩家叫牌类型向听数
        """
        call_hand_cards_len = len(hand_cards)
        # TODO: 碰杠数量(计算碰杠数)
        # 碰杠数 = 最多4个面子 - 手牌数 整除 3
        pong_gang_num = 4 - call_hand_cards_len // 3

        # TODO: 胡牌类型 -> 平胡，大队子
        self.round_over_the_worst_xts_by_hu_type = {
            HuPaiType.PING_HU: (self.most_hand_cards_len - 5 - 1) - pong_gang_num * 2,
            HuPaiType.DA_DUI_ZI: (self.most_hand_cards_len - 5 - 1) - pong_gang_num * 2,
        }

        # TODO: 胡牌类型 -> 七对，龙七对
        # 手牌十四张时
        if call_hand_cards_len == 13:
            # 最坏还需6个对子叫牌
            self.round_over_the_worst_xts_by_hu_type[HuPaiType.QI_DUI] = (self.most_hand_cards_len // 2) - 1
            self.round_over_the_worst_xts_by_hu_type[HuPaiType.LONG_QI_DUI] = (self.most_hand_cards_len // 2) - 1

        # TODO: 地龙七
        if call_hand_cards_len == 11:
            # 地龙七已经有一个碰（最坏还需5个对子叫牌）
            self.round_over_the_worst_xts_by_hu_type[HuPaiType.DI_LONG_QI] = (self.most_hand_cards_len // 2) - 2

    def calc_xts_by_normal_hu_type_old(self):
        """
        TODO: 计算胡牌向听数以及最优拆牌
        此接口用于计算机器人胡牌，机器人能胡则胡，胡牌效率较高
        机器人优先以向听数最小的胡牌牌型做牌并选择最优出牌
        """
        # args = [[单张], [对子], [刻子], [四张]]
        print("当前手牌和数量: ", self.__hand_cards, self.__hand_cards_len)
        args = self.calc_cards_list_by_count()

        # 计算各类作牌类型最优出牌
        # 卡牌、有效牌
        all_cards = []
        all_xts_yxp = []

        # 1.平胡
        xts1, best_cards1 = self.cal_xts_ping_hu(*args)
        print(f"平胡向听数: {xts1}, 最优出牌: {best_cards1}")
        print()
        all_cards += best_cards1
        all_xts_yxp.append((xts1, best_cards1))

        # 2.大对子
        xts2, best_cards2 = self.cal_xts_da_dui_zi(*args)
        all_cards += best_cards2
        all_xts_yxp.append((xts2, best_cards2))
        print(f"大对子向听数: {xts2}, 最优出牌: {best_cards2}")
        print()

        # 卡牌数量为14张时，判断小七对，龙七对
        if self.__hand_cards_len == 14:
            # 3.小七对
            xts3, best_cards3 = self.cal_xts_by_qi_dui(*args)
            all_cards += best_cards3
            all_xts_yxp.append((xts3, best_cards3))
            print(f"小七对向听数: {xts3}, 最优出牌: {best_cards3}")
            print()

            # 4.龙七对
            if len(args[2]) == 1:
                xts4, best_cards4 = self.cal_xts_long_qi_dui(*args)
                all_cards += best_cards4
                all_xts_yxp.append((xts4, best_cards4))
                print(f"龙七对向听数: {xts4}, 最优出牌: {best_cards4}")
                print()

        cards_to_map = self.cards_to_count(all_cards)
        best_card = max(cards_to_map, key=cards_to_map.get)

        if cards_to_map.get(best_card, 0) > 2:
            return best_card
        else:
            all_xts_yxp.sort(key=lambda x: x[0])
            for yxp_tup in all_xts_yxp:
                best_cards = yxp_tup[1]
                if best_cards:
                    return random.choice(best_cards)

            return -1

    def match_ping_hu(self, ting_cards, no_dian_pao=False, *args):
        """
        平胡出牌是否被大牌胡
        """
        if not no_dian_pao:
            return self.cal_xts_ping_hu(*args)
        return self.cal_xts_ping_hu_no_dian_pao(ting_cards, *args)

    def match_da_dui_zi(self, ting_cards, no_dian_pao=False, *args):
        """
        大队子出牌是否被大牌胡
        """
        if not no_dian_pao:
            return self.cal_xts_da_dui_zi(*args)
        return self.cal_xts_da_dui_zi_no_dian_pao(ting_cards, *args)

    def match_qi_dui(self, ting_cards, no_dian_pao=False, *args):
        """
        七对出牌是否被大牌胡
        """
        if not no_dian_pao:
            return self.cal_xts_by_qi_dui(*args)
        return self.cal_xts_by_qi_dui_no_dian_pao(ting_cards, *args)

    def match_long_qi_dui(self, ting_cards, no_dian_pao=False, *args):
        """
        龙七对出牌是否被大牌胡
        """
        if not no_dian_pao:
            return self.cal_xts_long_qi_dui(*args)
        return self.cal_xts_long_qi_dui_no_dian_pao(ting_cards, *args)

    def build_normal_types(self, ting_cards=None, no_dian_pao=False):
        """
        构建卡牌类型
        """
        # 听牌
        ting_cards = ting_cards or []
        # args = [[单张], [对子], [刻子], [四张]]
        args = self.calc_cards_list_by_count()
        print("当前手牌和数量: ", self.__hand_cards, self.__hand_cards_len)

        # 计算各类作牌类型最优出牌
        # 卡牌、有效牌
        all_cards = []
        all_xts_yxp = []

        # 1.平胡
        xts1, best_cards1 = self.match_ping_hu(ting_cards, no_dian_pao, *args)
        all_cards.extend(best_cards1)
        all_xts_yxp.append(("ph", xts1, best_cards1))
        print(f"平胡向听数: {xts1}, 最优出牌: {best_cards1}")
        print()

        # 卡牌数量为14，判断大队子、七对、龙七对
        if self.__hand_cards_len == 14:
            # 2.大对子: 3对+1刻 or 4对
            if len(args[1]) + len(args[2]) > self.ddz_flag_len or len(args[1]) > self.xqd_flag_len:
                xts2, best_cards2 = self.match_da_dui_zi(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards2)
                all_xts_yxp.append(("ddz", xts2, best_cards2))
                print(f"大对子向听数: {xts2}, 最优出牌: {best_cards2}")
                print()

            # 3.小七对
            if len(args[1]) > self.xqd_flag_len:
                xts3, best_cards3 = self.match_qi_dui(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards3)
                all_xts_yxp.append(("xqd", xts3, best_cards3))
                print(f"小七对向听数: {xts3}, 最优出牌: {best_cards3}")
                print()

            # 3.龙七对
            if len(args[1]) and len(args[2]) > self.ddz_flag_len:
                xts4, best_cards4 = self.match_long_qi_dui(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards4)
                all_xts_yxp.append(("lqd", xts4, best_cards4))
                print(f"龙七对向听数: {xts4}, 最优出牌: {best_cards4}")
                print()

        # 卡牌数量为11，判断大对
        elif self.__hand_cards_len == 11:
            # 大对子(处理11张大对子)
            # 额外处理(碰)后，是否能够组成大队子(3对+1刻 or 3对)
            if len(args[1]) + len(args[2]) > 2 or len(args[1]) > self.ddz_flag_len:
                xts5, best_cards5 = self.match_da_dui_zi(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards5)
                all_xts_yxp.append(("ddz", xts5, best_cards5))
                print(f"11张大对子向听数: {xts5}, 最优出牌: {best_cards5}")
                print()

        # 处理小于11张大对子
        elif self.__hand_cards_len < 11:
            # 处理存已碰杠后大对子
            if len(args[1]) + len(args[2]) > 2 or \
                    len(args[1]) + len(self.piles) > 2 or \
                    len(args[1]) + len(args[2]) + len(self.piles) > 3:
                xts6, best_cards6 = self.match_da_dui_zi(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards6)
                all_xts_yxp.append(("ddz", xts6, best_cards6))
                print(f"小于11张大对子向听数: {xts6}, 最优出牌: {best_cards6}")
                print()

        return all_cards, all_xts_yxp, best_cards1

    def build_qys_types(self, ting_cards=None, no_dian_pao=False):
        """
        构建清一色卡牌类型
        """
        ting_cards = ting_cards or []
        print("当前手牌和数量: ", self.__hand_cards, self.__hand_cards_len)
        args = self.calc_cards_list_by_count()
        all_cards = []
        all_xts_yxp = []

        # TODO: 计算卡牌构成的牌型(清一色: 平胡、小七对、大对子、龙七对)
        # 1.清一色: 平胡
        xts1, best_cards1 = self.match_ping_hu(ting_cards, no_dian_pao, *args)
        all_cards.extend(best_cards1)
        all_xts_yxp.append(("ph", xts1, best_cards1))
        print(f"清一色平胡向听数: {xts1}, 最优出牌: {best_cards1}")
        print()

        # 卡牌数量为14，判断清一色: 龙七对、大对子、小七对
        if self.__hand_cards_len == 14:
            # 2.清一色: 大对子(3对+1刻 or 4对)
            if len(args[1]) + len(args[2]) > self.ddz_flag_len or len(args[1]) > self.xqd_flag_len:
                xts2, best_cards2 = self.match_da_dui_zi(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards2)
                all_xts_yxp.append(("ddz", xts2, best_cards2))
                print(f"清一色大对子向听数: {xts2}, 最优出牌: {best_cards2}")
                print()

            # 3.清一色: 小七对
            if len(args[1]) > self.xqd_flag_len:
                xts3, best_cards3 = self.match_qi_dui(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards3)
                all_xts_yxp.append(("xqd", xts3, best_cards3))
                print(f"清一色小七对向听数: {xts3}, 最优出牌: {best_cards3}")
                print()

            # 4.清一色: 龙七对
            if len(args[1]) > self.ddz_flag_len and (len(args[2]) == 1 or len(args[-1]) == 1):
                xts4, best_cards4 = self.match_long_qi_dui(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards4)
                all_xts_yxp.append(("lqd", xts4, best_cards4))
                print(f"清龙七对向听数: {xts4}, 最优出牌: {best_cards4}")
                print()

        # 卡牌数量为11，判断清一色: 平胡、大对子
        elif self.__hand_cards_len == 11:
            # 判断碰牌后，是否能够组成清一色: 平胡、大对子
            if self.count_pg_types(self.__hand_cards[0] // 10):
                # 清一色: 平胡
                xts1, best_cards1 = self.match_ping_hu(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards1)
                all_xts_yxp.append(("ph", xts1, best_cards1))
                print(f"11张清一色平胡向听数: {xts1}, 最优出牌: {best_cards1}")
                print()

                # 清一色: 大对子(3对+1刻 or 3对)
                if len(args[1]) + len(args[2]) > 2 or len(args[1]) > 2:
                    xts6, best_cards6 = self.match_da_dui_zi(ting_cards, no_dian_pao, *args)
                    all_cards.extend(best_cards6)
                    all_xts_yxp.append(("ddz", xts6, best_cards6))
                    print(f"11张清大对子向听数: {xts6}, 最优出牌: {best_cards6}")
                    print()

        # 卡牌数量小于11张，判断清一色: 平胡、大对子
        elif self.__hand_cards_len < 11:
            # 判断碰、杠牌后，是否能够组成清一色: 平胡、大对子
            if self.count_pg_types(self.__hand_cards[0] // 10):
                # 清一色: 平胡
                xts1, best_cards1 = self.match_ping_hu(ting_cards, no_dian_pao, *args)
                all_cards.extend(best_cards1)
                all_xts_yxp.append(("ph", xts1, best_cards1))
                print(f"小于11张清一色平胡向听数: {xts1}, 最优出牌: {best_cards1}")
                print()

                # 清一色: 大对子
                if len(args[1]) + len(args[2]) > 2 or \
                        len(args[1]) + len(self.piles) > 2 or \
                        len(args[1]) + len(args[2]) + len(self.piles) > 3:
                    xts8, best_cards8 = self.match_da_dui_zi(ting_cards, no_dian_pao, *args)
                    all_cards.extend(best_cards8)
                    all_xts_yxp.append(("ddz", xts8, best_cards8))
                    print(f"小于11张清大对子向听数: {xts8}, 最优出牌: {best_cards8}")
                    print()

        return all_cards, all_xts_yxp, best_cards1

    def calc_max_hu_type(self, select_type, select_cards, ph_best_cards):
        """
        TODO: 寻找满足条件的最大胡牌类型
        计算除平胡外，向听数最小做牌类型
        """
        # 判断当前选择牌型是否在大牌牌型中，并且最优出牌不为空
        if select_type in self.max_hu_type and select_cards:
            # 计算平胡和大牌是否存在最优出牌，存在时，则优先选择打出
            eq_card = list(set(ph_best_cards) & set(select_cards))
            if eq_card:
                return random.choice(eq_card)
            # todo: 处理异常
            # 平胡与大牌不存在共同最优出牌时，则选择大牌最优出牌打出
            if not ph_best_cards and not select_cards:
                return random.choice(self.__hand_cards)
            return self.control_best_cards(ph_best_cards, select_cards)

    def select_hu_type(self, all_xts_yxp, ph_best_cards):
        """
        筛选胡牌类型作为本局牌型，此后按此牌型选择最优出牌
        """
        # 根据向听数大小排序，从小到大
        min_xts = sorted(all_xts_yxp, key=lambda x: x[1])

        # 最后一圈时，能叫嘴则先保证叫嘴
        if self.left_count < (self.modify_flag + 1) // 2:
            # 按最小向听数，选择最优出牌
            if not min_xts[0][2]:
                return random.choice(self.__hand_cards)
            # todo: 处理异常
            res_xts_yxp = self.calc_all_xts_best_cards(all_xts_yxp)
            if not min_xts[0][2] and not res_xts_yxp:
                return random.choice(self.__hand_cards)
            return self.control_best_cards(min_xts[0][2], res_xts_yxp)

        # 根据胡牌类型，由大到小选择做牌
        # 筛选掉平胡作大牌: 龙七对、大对子、小七对
        for idx, xts_result in enumerate(min_xts):
            # 按最小向听数出牌, 则直接返回
            # 处理异常卡牌数据，防止程序中断
            if min_xts[idx][1] < 0:
                return random.choice(ph_best_cards)

            # TODO: 向听数不为0时，且当前做牌不为平胡牌型，则按大牌方向打
            if min_xts[idx][1] < self.ddz_flag_len and min_xts[idx][0] != 'ph':
                return self.calc_max_hu_type(min_xts[idx][0], min_xts[idx][2], ph_best_cards)

        # 按最小向听数，选择最优出牌
        if not min_xts[0][2]:
            return random.choice(self.__hand_cards)
        # todo: 处理异常
        res_xts_yxp = self.calc_all_xts_best_cards(all_xts_yxp)
        if not min_xts[0][2] and not res_xts_yxp:
            return random.choice(self.__hand_cards)
        return self.control_best_cards(min_xts[0][2], res_xts_yxp)

    def count_pg_types(self, card_type):
        """
        判断当前手牌与碰杠牌是否为同一种花色
        """
        pg_cards = sum([pile[1:-1] for pile in self.piles], [])
        count_type = list(self.calc_same_card_type(pg_cards).keys())
        if len(count_type) == 1:
            if count_type[0] == card_type:
                return pg_cards
        return []

    @staticmethod
    def calc_same_card_type(cards):
        """
        计算当前手牌花色情况
        """
        cards_type = defaultdict(list)
        for card in cards:
            # 筛选掉万能牌，万能牌不参与碰/杠
            if card == CardType.LAI_ZI:
                continue
            # 万能牌可作为清一色牌型来进行组合
            tmp_type = card // 10
            cards_type[tmp_type].append(card)
        return cards_type

    def count_xqd_nums(self):
        """
        计算当前对子数量
        """
        # 当前手牌数量不等于13
        if self.__hand_cards_len != 13:
            return False

        # 统计牌型小七对对子数量
        count = 0
        hand_dict = self.cards_to_count(self.__hand_cards)
        for cards, nums in hand_dict.items():
            if nums == 2:
                count += 1

        # 判断小七对，对子数量
        # 对子数量大于4时，则不再进行碰牌
        return count if count > 4 else False

    def do_qys_cards(self, cards_type):
        """
        做清一色大牌
        """
        # 计算是否符合清一色出牌条件(碰、杠或未碰杠)
        for card_type, card_values in cards_type.items():
            # 无碰杠时，仅计算同一花色
            if not self.piles and self.left_count > self.modify_flag:
                if len(card_values) > self.qys_flag_len:
                    print("当前手牌: ", self.__hand_cards, self.__hand_cards_len)
                    print("%@## 当前同一花色牌: ", card_values, len(card_values))
                    print("%@### 开始做清一色牌型，将其他类型卡牌去掉")

                    # 选择其他花色打出，构成清一色牌型
                    play_cards = list(set(self.__hand_cards).difference(set(card_values)))
                    return random.choice(play_cards)

            # 存在碰杠时，计算同一花色是否碰杠过
            elif self.piles and self.left_count > self.modify_flag:
                action_cards = self.count_pg_types(card_type)
                if action_cards and len(card_values) + len(action_cards) > self.qys_flag_len:
                    print("当前手牌: ", self.__hand_cards, self.__hand_cards_len)
                    print("%@##@% 当前同一花色牌碰杠后: ", card_values, len(card_values))
                    print("%@####% 开始做清一色牌型，将其他类型卡牌去掉")

                    # 选择其他花色打出, 构成清一色牌型
                    play_cards = list(set(self.__hand_cards).difference(set(card_values)))
                    return random.choice(play_cards)

    def calc_others_player_qys(self, others_cards_and_piles):
        """
        TODO: 计算其他两位玩家是否与自己作同一花色牌型
        """
        # 1.判断当前玩家卡牌清一色情况(包含碰、杠)
        self_qys_type = False
        self_cards_type = self.calc_qing_yi_se()
        for self_type, self_cards in self_cards_type.items():
            pg_cards = self.count_pg_types(self_type)
            # 计算碰杠清一色卡牌是否符合条件
            # 符合条件则直接跳出
            if pg_cards:
                if len(pg_cards) + len(self_cards) > self.qys_flag_len + 1:
                    self_qys_type = self_type
                    break
            # 计算不存在碰杠时，清一色卡牌是否符合条件
            # 符合条件则直接跳出
            if len(self_cards) > self.qys_flag_len:
                self_qys_type = self_type
                break

        # 2.与其他两位玩家手牌进行比较，判断是否存在相同花色卡牌(清一色)
        if len(list(self_cards_type.keys())) == 1 or self_qys_type:
            for cards_piles in others_cards_and_piles:
                pg_cards = sum([pile[1:-1] for pile in cards_piles[-1]], [])
                pg_cards_type = self.calc_same_card_type(pg_cards)
                hd_cards_type = self.calc_same_card_type(cards_piles[0])

                # 计算其他玩家碰杠后是否存在清一色
                if pg_cards and self_qys_type:
                    hd_cards = hd_cards_type[self_qys_type]
                    if hd_cards and list(pg_cards_type.keys())[0] == self_qys_type:
                        if len(pg_cards) + len(hd_cards) > self.qys_flag_len + 1:
                            return hd_cards_type, False

                # 计算其他玩家无碰杠时是否为清一色
                for hd_type, hd_cards in hd_cards_type.items():
                    if hd_type == self_qys_type and len(hd_cards) > self.qys_flag_len:
                        return hd_cards_type, False

        # 当前牌型可打清一色，其他玩家无相同清一色牌型
        if self_qys_type or len(list(self_cards_type.keys())) == 1:
            return self_cards_type, self_qys_type

        return {}, False

    def calc_xts_by_normal_hu_type(self):
        """
        TODO: 计算胡牌向听数以及最优拆牌，有一定概率做大牌
        此接口用于计算机器人胡牌，机器人能胡则胡，胡牌效率一般
        机器人优先以向听数最小的胡牌牌型做牌并选择选择最优出牌
        """
        # 构建卡牌类型及对应的卡牌计算
        all_cards, all_xts_yxp, best_cards1 = self.build_normal_types()

        # 仅组成单个牌型
        if len(all_xts_yxp) == 1:
            # 无最优出牌，出那一张都被必胡，则从手牌随机选择可出卡牌
            if not all_xts_yxp[0][2]:
                return random.choice(self.__hand_cards)
            # 向听数小于0时，当前手牌形成胡牌
            # 选择牌盒中剩余卡牌数量最多的卡牌
            if all_xts_yxp[0][1] < 0:
                return random.choice(best_cards1)
            return random.choice(all_xts_yxp[0][2])

        # 至少组成两个及以上牌型
        min_xts = sorted(all_xts_yxp, key=lambda x: x[1])
        for idx, xts_result in enumerate(min_xts):
            # 向听数小于0时，当前手牌形成胡牌
            # 选择牌盒中剩余卡牌数量最多的卡牌
            if min_xts[idx][1] < 0:
                return random.choice(best_cards1)
            if min_xts[idx][0] == 'ph':
                continue
            # 添加做大牌概率
            if min_xts[idx][1] < 2:
                # 选择平胡和大牌存在的相同卡牌为最优出牌
                eq_card = list(set(min_xts[idx][2]) & set(best_cards1))
                if eq_card:
                    return random.choice(eq_card)
                return random.choice(min_xts[idx][2])

            # TODO: 此逻辑利于构建向听数小于2的大牌
            # 选择平胡和大牌存在的相同卡牌为最优出牌
            eq_card = list(set(all_cards) & set(best_cards1))
            if eq_card:
                return random.choice(eq_card)

            # 平胡与大牌不存在共同最优出牌时，则选择大牌最优出牌打出
            return random.choice(min_xts[0][2])

        # 上述条件不满足时，按最小向听数出牌, 则直接返回
        return random.choice(min_xts[0][2])

    def calc_xts_by_best_hu_type(self):
        """
        TODO: 计算分数最高的胡牌类型，优先级别按: 清一色、龙七对、大对子、小七对
        机器人胡牌效率较低，优先考虑作大牌(清一色、龙七对，大对子、小七对)，存在出牌会被点炮(大牌胡)
        """
        all_cards, all_xts_yxp, best_cards1 = self.build_normal_types()
        # 仅有一种可选牌型时，则直接按此牌型最优出牌打出
        if len(all_xts_yxp) == 1:
            # 无最优出牌，出那一张都被必胡，则从手牌随机选择可出卡牌
            if not all_xts_yxp[0][2]:
                return random.choice(self.__hand_cards)
            # 处理异常卡牌数据，防止程序异常中断
            if all_xts_yxp[0][1] < 0:
                return random.choice(best_cards1)
            res_xts_yxp = self.calc_all_xts_best_cards(all_xts_yxp)
            # todo: 处理异常
            if not best_cards1 and not res_xts_yxp:
                return random.choice(self.__hand_cards)
            return self.control_best_cards(best_cards1, res_xts_yxp)
        # 存在多种牌型
        # 1.计算向听数相差2范围内牌型打出
        # 2.都不满足，选最小向听数出牌，包括0向听
        elif len(all_xts_yxp) > 1:
            return self.select_hu_type(all_xts_yxp, best_cards1)

    @staticmethod
    def calc_all_xts_best_cards(all_xts_yxp):
        """
        计算所有有效出牌
        """
        all_best_cards = []
        for xts_yxp in all_xts_yxp:
            all_best_cards.extend(xts_yxp[2])
        return all_best_cards

    @staticmethod
    def calc_all_xts_best_cards_bak(all_xts_yxp):
        """
        # todo: 代码执行效率优化
        计算所有有效牌
        """
        return sum([xts_yxp[2] for xts_yxp in all_xts_yxp], [])

    @staticmethod
    def control_best_cards(best_cards, all_xts_yxp):
        """
        控制机器人根据向听数选最佳的出牌
        """
        # 不存在最佳出牌或无最佳出牌时，则不通过概率来选取出牌
        if not best_cards or not all_xts_yxp:
            return random.choice(best_cards + all_xts_yxp)

        # 条件满足时，则根据设置概率来选择出牌
        best_card = random.choice(best_cards)
        other_cards = random.choice(all_xts_yxp)
        res = random_choice_num([best_card, other_cards], [0.7, 0.3])
        return int(res)

    def calc_xts_by_max_hu_type(self, ting_cards=None, others_cards_and_piles=None):
        """
        TODO: 根据向听数构建大牌胡牌类型，清一色 or 非清一色(平胡、七对，大对，龙七对)
            1.清一色胡牌类型
            2.非清一色胡牌类型
            3.清一色: 同一花色数量 > 9
            4.七对: 对子数量 > 4
            5.大对: 对子数量 + 刻子数量 > 3
        """
        # 判断清一色牌型是否符合条件，清一色牌型分为玩家未进行过碰、杠或进行过碰、杠
        ting_cards = ting_cards or []
        others_cards_and_piles = others_cards_and_piles or []
        cards_type, qys_flag = self.calc_others_player_qys(others_cards_and_piles)

        # TODO: 计算清一色牌型及最优出牌
        # 判断未碰杠或杠牌后当前所做牌型是否为清一色
        if len(list(cards_type.keys())) == 1 and qys_flag:
            # 计算清一色牌型
            all_cards, all_xts_yxp, best_cards1 = self.build_qys_types(ting_cards)

            # TODO: 计算最佳牌型和最优出牌
            # 仅有一种可选牌型时，则直接按此牌型最优出牌打出
            if len(all_xts_yxp) == 1:
                # 无最优出牌，出那一张都被必胡，则从手牌随机选择可出卡牌
                if not all_xts_yxp[0][2]:
                    return random.choice(self.__hand_cards)
                # 处理异常卡牌数据，当前手牌形成了胡牌
                # 此处逻辑通常不会进入，形成胡牌时，则从随机打出一张
                if all_xts_yxp[0][1] < 0:
                    return random.choice(best_cards1)
                # todo: 处理异常
                res_xts_yxp = self.calc_all_xts_best_cards(all_xts_yxp)
                if not best_cards1 and not res_xts_yxp:
                    return random.choice(self.__hand_cards)
                return self.control_best_cards(best_cards1, res_xts_yxp)

            # 存在多种牌型
            # 1.计算向听数相差2范围内牌型打出
            # 2.都不满足，选最小向听数出牌，包括0向听
            elif len(all_xts_yxp) > 1:
                return self.select_hu_type(all_xts_yxp, best_cards1)

            # 无满足条件牌型
            return self.calc_xts_by_best_hu_type()

        # TODO: 清一色做牌
        # 大牌清一色条件，同一类花色必须大于10
        # 制作清一色牌型，选择随机不是清一色卡牌打出，凑成同一花色类型卡牌
        elif qys_flag:
            result = self.do_qys_cards(cards_type)
            if not result:
                # TODO: 上面条件不满足，则选择最佳胡牌牌型
                # 此接口在上面清一色牌型不能做牌时调用，选择最佳牌型和最优出牌
                return self.calc_xts_by_best_hu_type()
            return result

        # TODO: 上面条件不满足(不是清一色，也不能做清一色牌型)，则选择最佳胡牌牌型
        # 此接口在上面清一色牌型不能做牌时调用，选择最佳牌型和最优出牌
        else:
            return self.calc_xts_by_best_hu_type()

    def calc_hu_type_no_dian_pao(self, ting_cards, no_dian_pao):
        """
        TODO: 计算出牌是否放炮对手大牌型(除掉平胡)
        此接口出牌时，不包含对手大牌所需胡牌(听牌)，胡牌类型不包含平胡
        机器人胡牌效率较低，优先考虑作大牌(清一色、龙七对，大对子、小七对)，不存在出牌会被点炮(大牌胡)
        """
        all_cards, all_xts_yxp, best_cards1 = self.build_normal_types(ting_cards, no_dian_pao)

        # 只能组成一种类型时，直接按此类型打
        if len(all_xts_yxp) == 1:
            if all_xts_yxp[0][1] < 0:
                return random.choice(best_cards1)
            return random.choice(all_xts_yxp[0][2])

        # 存在多种牌型
        # 1.计算向听数相差2范围内牌型打出
        # 2.都不满足，选最小向听数出牌，包括0向听
        elif len(all_xts_yxp) > 1:
            return self.select_hu_type(all_xts_yxp, best_cards1)

    def calc_xts_by_max_hu_type_no_dian_pao(self, ting_list=None, others_cards_and_piles=None, no_dian_pao=True):
        """
        TODO: 计算出牌是否放炮对手大牌(不包含平胡，如，小七对，大对子，龙七对，清一色...)
            1.判断是否叫牌
            2.判断出牌是否点炮
            3.点炮类型为平胡时，则直接打，如果点炮为大牌，则拆掉手牌出，可流局或不听牌，也不点炮
        """
        # 计算真人非平胡类型听牌(小七对，大对子，龙七对，清一色)
        ting_list = ting_list or []
        others_cards_and_piles = others_cards_and_piles or []
        ting_cards = self.get_round_over_call_type_and_ting_cards(ting_list, others_cards_and_piles)
        # 真人玩家是否进行过天听或锁🔒牌
        if ting_list:
            ting_cards.extend(ting_list)

        # 判断对局中卡牌是否符合组成清一色牌条件
        # 判断其他玩家是否和自己打相同的清一色卡牌类型
        cards_type, qys_flag = self.calc_others_player_qys(others_cards_and_piles)
        # 卡牌全为同一花色，则计算最优出牌(清一色)
        if len(list(cards_type.keys())) == 1 and qys_flag:
            # 计算清一色牌型
            all_cards, all_xts_yxp, best_cards1 = self.build_qys_types(ting_cards, no_dian_pao)

            # 仅存一种牌型，直接打牌即可
            if len(all_xts_yxp) == 1:
                if all_xts_yxp[0][1] < 0:
                    return random.choice(best_cards1)
                return random.choice(all_xts_yxp[0][2])

            # 存在多种牌型
            # 1.计算向听数相差2范围内牌型打出
            # 2.都不满足，选最小向听数出牌，包括0向听
            elif len(all_xts_yxp) > 1:
                return self.select_hu_type(all_xts_yxp, best_cards1)

            # 无满足条件牌型
            else:
                return self.calc_hu_type_no_dian_pao(ting_cards, no_dian_pao)

        # TODO: 清一色做牌, 对局中，作相同花色卡牌，只能存在一位玩家
        # True能作清一色，False则不能作清一色
        # 大牌清一色条件，同一类花色必须大于10
        # 制作清一色牌型，选择随机不是清一色卡牌打出，凑成同一花色类型卡牌
        elif qys_flag:
            result = self.do_qys_cards(cards_type)
            if not result:
                # TODO: 上面条件不满足，则选择最佳胡牌牌型
                # 此接口在上面清一色牌型不能做牌时调用，选择最佳牌型和最优出牌
                return self.calc_hu_type_no_dian_pao(ting_cards, no_dian_pao)
            return result

        else:
            # TODO: 与清一色牌型无关时
            return self.calc_hu_type_no_dian_pao(ting_cards, no_dian_pao)

    def cal_xts_ping_hu(self, *args, need_mz=None):
        """
        计算平胡向听数判断平胡
        """
        # 胡牌类型(平胡)
        # 参数解析(单张、两张、三张、四张)
        one_list, two_list, three_list, four_list = args

        # 面子(顺子、刻子)
        # 平胡: 1个对子 + 4个面子 -> (2 + 3 x 4) = 14
        need_mz = need_mz or self.__hand_cards_len // 3
        need_heap = need_mz if need_mz > 0 else 0  # 需要多少堆
        optimal_path = []

        record_lowest_xts = 8  # 最小的向听数（仅平胡）用于全局记录最低的向听数
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.PING_HU)  # 当前牌的最坏向听数（减去了碰杠）
        jiang_list = two_list + three_list + four_list  # 不添单张(计算大于两张的卡牌)

        # if not two_list:
        jiang_list += one_list
        for pair in jiang_list:  # 列表相加得到新实例
            new_hand_cards = self.__hand_cards[:]
            self.remove_by_value(new_hand_cards, pair, 2)  # 减去对子（平胡只能有一个将）
            # 当做将的对子不是从刻子中取的，则先拆刻子
            if pair in one_list:
                split_path = [[pair]]
            else:
                split_path = [[pair] * 2]

            # print(f"对子(将牌): {pair}, 所需搭子数: {need_heap}")

            def optimal_split_cards(hand_cards):
                """
                TODO: 最优拆牌
                params: new_hand_cards  去掉对子/刻子的手牌
                params: optimal_path 最优路径
                params: all_split_cards 所有组合
                params: need_heap  需要堆数
                """
                nonlocal self
                nonlocal optimal_path
                nonlocal record_lowest_xts
                nonlocal need_heap
                nonlocal split_path

                # 减去 对子/刻子 后的所有搭子
                # todo: 遍历找出顺子
                all_split_cards = []
                hand_cards_copy = hand_cards[:]
                count = 0
                extra_shun = []
                while hand_cards_copy:
                    # 此操作为了找出所有的顺子（包含小顺）
                    single_cards = sorted(list(set(hand_cards_copy)))
                    for sc in single_cards:
                        hand_cards_copy.remove(sc)

                    # 计算所有顺子(三连顺，二连顺，咔张)
                    all_shun = self.gen_serial_moves(single_cards)
                    if count > 0:
                        # all_shun = [shun for shun in all_shun if len(shun) == 3]
                        extra_shun.extend(all_shun)
                    if not all_shun:
                        break
                    all_split_cards.extend(all_shun)
                    count += 1

                # 计算每一张手牌对应的数量(统计当前卡牌数量)
                new_cards_to_count = self.calc_cards_to_count(hand_cards)
                if not all_split_cards:
                    if record_lowest_xts < the_worst_xts:
                        return
                    record_lowest_xts = the_worst_xts
                    res_split_cards = split_path[:]
                    for card, count in new_cards_to_count.items():
                        count > 0 and res_split_cards.append([card] * count)
                    optimal_path.append(res_split_cards)
                else:
                    all_can_comb_shun_cards = []
                    for shun in all_split_cards:
                        all_can_comb_shun_cards.extend(shun)
                    all_can_comb_shun_cards = list(set(all_can_comb_shun_cards))
                    if extra_shun:
                        es_shun_cards = []
                        for es in extra_shun:
                            es_shun_cards.extend(es)
                        all_can_comb_shun_cards.extend(list(set(es_shun_cards)))

                    new_cards_to_count_copy = new_cards_to_count.copy()
                    for s in all_can_comb_shun_cards:
                        new_cards_to_count_copy[s] -= 1
                    extra_comb = []

                    # 计算刻子
                    for card, count in new_cards_to_count_copy.items():
                        if count > 0:
                            # 刻子统一在下面添加，此处可能原本刻子被拆了
                            if count == 3 and card in three_list:
                                continue
                            extra_comb.append([card] * count)

                    extra_pair = []
                    for two in two_list:
                        if two == pair:
                            continue
                        two_val = MoveGenerator.get_value(two)
                        if two_val == 1:
                            if new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 2:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(
                                    two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 8:
                            if new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(
                                    two - 1) or new_cards_to_count_copy.get(two - 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 9:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(two - 2):
                                extra_pair.append([two] * 2)
                        else:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(two - 2) or \
                                    new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)

                    all_split_cards.extend(extra_pair)
                    ke_zi = [[card] * 3 for card in three_list + four_list if card != pair]  # 刻子也添加进去
                    all_split_cards.extend(ke_zi)
                    all_split_cards.extend(extra_comb)

                    # 先拆顺子
                    # TODO: 再计算搭子(二连顺、间隔顺)
                    curr_heap_idx = 0  # 当前堆的索引
                    record_comb = ...
                    all_comb = itertools.combinations(range(len(all_split_cards)), need_heap)
                    # all_comb_list = list(all_comb)
                    # print("所有组合长度：", len(all_comb_list), all_comb_list)
                    for comb in all_comb:
                        if comb[:curr_heap_idx + 1] == record_comb:
                            continue
                        # 统计每一张手牌数量
                        cards_to_count_copy = new_cards_to_count.copy()
                        comb_list = []
                        curr_heap_idx = 0
                        record_comb = ...
                        flag = True
                        # 根据所需的搭子数，计算搭子
                        for i in range(need_heap):
                            curr_heap_idx = i
                            one_comb = all_split_cards[comb[i]]
                            for c in one_comb:
                                if cards_to_count_copy.get(c) <= 0:
                                    flag = False
                                    record_comb = comb[:i + 1]
                                    break
                                cards_to_count_copy[c] -= 1
                            if not flag:
                                comb_list.clear()
                                break
                            comb_list.append(one_comb)

                        if comb_list:
                            res_split_cards = []
                            res_split_cards.extend(split_path)
                            res_split_cards.extend(comb_list)

                            # 平胡拆牌后，计算向听数 注意：res_split_cards就是根据需要堆数的组合，所以不用考虑多个对子的向听数问题
                            xts = the_worst_xts
                            for sc in res_split_cards:
                                comb_type = MoveGenerator.judge_cards_type(sc)
                                comb_xts = COMB_REMOVE_XTS.get(comb_type, 0)
                                xts -= comb_xts

                            for card, count in cards_to_count_copy.items():
                                count > 0 and res_split_cards.append([card] * count)

                            if xts < record_lowest_xts:
                                optimal_path.clear()
                                optimal_path.append(res_split_cards)
                                record_lowest_xts = xts

                            elif xts == record_lowest_xts:
                                optimal_path.append(res_split_cards)

            optimal_split_cards(new_hand_cards)

        # 去重
        deduplicate = []
        for op in optimal_path:
            if op in deduplicate:
                continue
            deduplicate.append(op)

        best_cards = self.cal_yxp_by_ping_hu(deduplicate, need_heap, record_lowest_xts)

        return record_lowest_xts, best_cards

    def cal_xts_ping_hu_no_dian_pao(self, ting_cards, *args, need_mz=None):
        """
        计算平胡向听数，出牌不被点炮
        """
        # 胡牌类型(平胡)
        # 参数解析(单张、两张、三张、四张)
        one_list, two_list, three_list, four_list = args

        # 面子(顺子、刻子)
        # 平胡: 1个对子 + 4个面子 -> (2 + 3 x 4) = 14
        need_mz = need_mz or self.__hand_cards_len // 3
        need_heap = need_mz if need_mz > 0 else 0  # 需要多少堆
        optimal_path = []

        record_lowest_xts = 8  # 最小的向听数（仅平胡）用于全局记录最低的向听数
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.PING_HU)  # 当前牌的最坏向听数（减去了碰杠）
        jiang_list = two_list + three_list + four_list  # 不添单张(计算大于两张的卡牌)

        # if not two_list:
        jiang_list += one_list
        for pair in jiang_list:  # 列表相加得到新实例
            new_hand_cards = self.__hand_cards[:]
            self.remove_by_value(new_hand_cards, pair, 2)  # 减去对子（平胡只能有一个将）
            # 当做将的对子不是从刻子中取的，则先拆刻子
            if pair in one_list:
                split_path = [[pair]]
            else:
                split_path = [[pair] * 2]

            # print(f"对子(将牌): {pair}, 所需搭子数: {need_heap}")

            def optimal_split_cards(hand_cards):
                """
                TODO: 最优拆牌
                params: new_hand_cards  去掉对子/刻子的手牌
                params: optimal_path 最优路径
                params: all_split_cards 所有组合
                params: need_heap  需要堆数
                """
                nonlocal self
                nonlocal optimal_path
                nonlocal record_lowest_xts
                nonlocal need_heap
                nonlocal split_path

                # 减去 对子/刻子 后的所有搭子
                # todo: 遍历找出顺子
                all_split_cards = []
                hand_cards_copy = hand_cards[:]
                count = 0
                extra_shun = []
                while hand_cards_copy:
                    # 此操作为了找出所有的顺子（包含小顺）
                    single_cards = sorted(list(set(hand_cards_copy)))
                    for sc in single_cards:
                        hand_cards_copy.remove(sc)

                    # 计算所有顺子(三连顺，二连顺，咔张)
                    all_shun = self.gen_serial_moves(single_cards)
                    if count > 0:
                        # all_shun = [shun for shun in all_shun if len(shun) == 3]
                        extra_shun.extend(all_shun)
                    if not all_shun:
                        break
                    all_split_cards.extend(all_shun)
                    count += 1

                # 计算每一张手牌对应的数量(统计当前卡牌数量)
                new_cards_to_count = self.calc_cards_to_count(hand_cards)
                if not all_split_cards:
                    if record_lowest_xts < the_worst_xts:
                        return
                    record_lowest_xts = the_worst_xts
                    res_split_cards = split_path[:]
                    for card, count in new_cards_to_count.items():
                        count > 0 and res_split_cards.append([card] * count)
                    optimal_path.append(res_split_cards)
                else:
                    all_can_comb_shun_cards = []
                    for shun in all_split_cards:
                        all_can_comb_shun_cards.extend(shun)
                    all_can_comb_shun_cards = list(set(all_can_comb_shun_cards))
                    if extra_shun:
                        es_shun_cards = []
                        for es in extra_shun:
                            es_shun_cards.extend(es)
                        all_can_comb_shun_cards.extend(list(set(es_shun_cards)))

                    new_cards_to_count_copy = new_cards_to_count.copy()
                    for s in all_can_comb_shun_cards:
                        new_cards_to_count_copy[s] -= 1
                    extra_comb = []

                    # 计算刻子
                    for card, count in new_cards_to_count_copy.items():
                        if count > 0:
                            # 刻子统一在下面添加，此处可能原本刻子被拆了
                            if count == 3 and card in three_list:
                                continue
                            extra_comb.append([card] * count)

                    extra_pair = []
                    for two in two_list:
                        if two == pair:
                            continue
                        two_val = MoveGenerator.get_value(two)
                        if two_val == 1:
                            if new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 2:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(
                                    two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 8:
                            if new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(
                                    two - 1) or new_cards_to_count_copy.get(two - 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 9:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(two - 2):
                                extra_pair.append([two] * 2)
                        else:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(two - 2) or \
                                    new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)

                    all_split_cards.extend(extra_pair)
                    ke_zi = [[card] * 3 for card in three_list + four_list if card != pair]  # 刻子也添加进去
                    all_split_cards.extend(ke_zi)
                    all_split_cards.extend(extra_comb)

                    # 先拆顺子
                    # TODO: 再计算搭子(二连顺、间隔顺)
                    curr_heap_idx = 0  # 当前堆的索引
                    record_comb = ...
                    all_comb = itertools.combinations(range(len(all_split_cards)), need_heap)
                    # all_comb_list = list(all_comb)
                    # print("所有组合长度：", len(all_comb_list), all_comb_list)
                    for comb in all_comb:
                        if comb[:curr_heap_idx + 1] == record_comb:
                            continue
                        # 统计每一张手牌数量
                        cards_to_count_copy = new_cards_to_count.copy()
                        comb_list = []
                        curr_heap_idx = 0
                        record_comb = ...
                        flag = True
                        # 根据所需的搭子数，计算搭子
                        for i in range(need_heap):
                            curr_heap_idx = i
                            one_comb = all_split_cards[comb[i]]
                            for c in one_comb:
                                if cards_to_count_copy.get(c) <= 0:
                                    flag = False
                                    record_comb = comb[:i + 1]
                                    break
                                cards_to_count_copy[c] -= 1
                            if not flag:
                                comb_list.clear()
                                break
                            comb_list.append(one_comb)

                        if comb_list:
                            res_split_cards = []
                            res_split_cards.extend(split_path)
                            res_split_cards.extend(comb_list)

                            # 平胡拆牌后，计算向听数 注意：res_split_cards就是根据需要堆数的组合，所以不用考虑多个对子的向听数问题
                            xts = the_worst_xts
                            for sc in res_split_cards:
                                comb_type = MoveGenerator.judge_cards_type(sc)
                                comb_xts = COMB_REMOVE_XTS.get(comb_type, 0)
                                xts -= comb_xts

                            for card, count in cards_to_count_copy.items():
                                count > 0 and res_split_cards.append([card] * count)

                            if xts < record_lowest_xts:
                                optimal_path.clear()
                                optimal_path.append(res_split_cards)
                                record_lowest_xts = xts
                            elif xts == record_lowest_xts:
                                optimal_path.append(res_split_cards)

            optimal_split_cards(new_hand_cards)

        # 去重
        deduplicate = []
        for op in optimal_path:
            if op in deduplicate:
                continue
            deduplicate.append(op)

        best_cards = self.cal_yxp_by_ping_hu_no_dian_pao(deduplicate, need_heap, record_lowest_xts, ting_cards)

        # 判断全部最优出牌是否都不能打出
        # 当所有最优出牌不可选时，则重新计算选择出牌，避免被大牌胡
        if not best_cards:
            # 卡牌选择顺序: 单张 -> 间隔、二连 -> 三连
            # 先单张、间隔、二连顺优先选择出牌
            new_one_list = list(set(one_list) - set(ting_cards))
            if new_one_list:
                # 只存在一张卡牌时，优先选择此牌作为当前出牌
                if len(new_one_list) == 1:
                    return record_lowest_xts, new_one_list

                # 存在两张卡牌时，优先选择间隔卡牌作为当前出牌，再考虑二连顺
                elif len(new_one_list) == 2:
                    if self.calc_gap_cards(new_one_list[0], new_one_list[-1]):
                        return record_lowest_xts + 1, new_one_list
                    return record_lowest_xts + 1, new_one_list

                # 当单张、间隔、二连顺都不存在优先选择出牌时，则从三连中选择当前出牌
                else:
                    one_cards, two_cards, three_cards = self.unpack_deduplicate_cards(deduplicate, ting_cards)
                    # 计算单张
                    if one_cards:
                        return record_lowest_xts, one_cards
                    # 间隔、二连顺
                    elif two_cards:
                        return record_lowest_xts + 1, two_cards
                    # 刻子(三连顺)
                    return record_lowest_xts + 1, three_cards

            # 上面无满足条件时，则从对子中选择当前出牌
            new_two_list = list(set(two_list) - set(ting_cards))
            if new_two_list:
                return record_lowest_xts + 1, new_two_list

            # 上面无满足条件时，则从刻子(三张相同)中选择当前出牌
            new_three_list = list(set(three_list) - set(ting_cards))
            if new_three_list:
                return record_lowest_xts + 1, new_two_list

            # 无最优出牌，出那一张都被必胡，则从手牌随机选择可出卡牌
            if not best_cards:
                return record_lowest_xts, self.__hand_cards

        return record_lowest_xts, best_cards

    def get_round_over_by_ping_hu_ting_cards(self, hand_cards, *args):
        """
        计算一轮结束后，真人玩家叫牌类型为平胡所听卡牌
        """
        # 胡牌类型(平胡)
        # 参数解析(单张、两张、三张、四张)
        one_list, two_list, three_list, four_list = args

        # 面子(顺子、刻子)
        # 平胡: 1个对子 + 4个面子 -> (2 + 3 x 4) = 14
        need_mz = len(hand_cards) // 3
        need_heap = need_mz if need_mz > 0 else 0  # 需要多少堆
        optimal_path = []

        record_lowest_xts = 8  # 最小的向听数（仅平胡）用于全局记录最低的向听数
        the_worst_xts = self.round_over_the_worst_xts_by_hu_type.get(HuPaiType.PING_HU)  # 当前牌的最坏向听数（减去了碰杠）
        jiang_list = two_list + three_list + four_list  # 不添单张(计算大于两张的卡牌)

        # if not two_list:
        jiang_list += one_list
        for pair in jiang_list:  # 列表相加得到新实例
            new_hand_cards = hand_cards[:]
            self.remove_by_value(new_hand_cards, pair, 2)  # 减去对子（平胡只能有一个将）
            # 当做将的对子不是从刻子中取的，则先拆刻子
            if pair in one_list:
                split_path = [[pair]]
            else:
                split_path = [[pair] * 2]

            # print(f"对子(将牌): {pair}, 所需搭子数: {need_heap}")

            def optimal_split_cards(hand_cards):
                """
                TODO: 最优拆牌
                params: new_hand_cards  去掉对子/刻子的手牌
                params: optimal_path 最优路径
                params: all_split_cards 所有组合
                params: need_heap  需要堆数
                """
                nonlocal self
                nonlocal optimal_path
                nonlocal record_lowest_xts
                nonlocal need_heap
                nonlocal split_path

                # 减去 对子/刻子 后的所有搭子
                # todo: 遍历找出顺子
                all_split_cards = []
                hand_cards_copy = hand_cards[:]
                count = 0
                extra_shun = []
                while hand_cards_copy:
                    # 此操作为了找出所有的顺子（包含小顺）
                    single_cards = sorted(list(set(hand_cards_copy)))
                    for sc in single_cards:
                        hand_cards_copy.remove(sc)

                    # 计算所有顺子(三连顺，二连顺，咔张)
                    all_shun = self.gen_serial_moves(single_cards)
                    if count > 0:
                        # all_shun = [shun for shun in all_shun if len(shun) == 3]
                        extra_shun.extend(all_shun)
                    if not all_shun:
                        break
                    all_split_cards.extend(all_shun)
                    count += 1

                # 计算每一张手牌对应的数量(统计当前卡牌数量)
                new_cards_to_count = self.cards_to_count(hand_cards)
                if not all_split_cards:
                    if record_lowest_xts < the_worst_xts:
                        return
                    record_lowest_xts = the_worst_xts
                    res_split_cards = split_path[:]
                    for card, count in new_cards_to_count.items():
                        count > 0 and res_split_cards.append([card] * count)
                    optimal_path.append(res_split_cards)
                else:
                    all_can_comb_shun_cards = []
                    for shun in all_split_cards:
                        all_can_comb_shun_cards.extend(shun)
                    all_can_comb_shun_cards = list(set(all_can_comb_shun_cards))
                    if extra_shun:
                        es_shun_cards = []
                        for es in extra_shun:
                            es_shun_cards.extend(es)
                        all_can_comb_shun_cards.extend(list(set(es_shun_cards)))

                    new_cards_to_count_copy = new_cards_to_count.copy()
                    for s in all_can_comb_shun_cards:
                        new_cards_to_count_copy[s] -= 1
                    extra_comb = []

                    # 计算刻子
                    for card, count in new_cards_to_count_copy.items():
                        if count > 0:
                            # 刻子统一在下面添加，此处可能原本刻子被拆了
                            if count == 3 and card in three_list:
                                continue
                            extra_comb.append([card] * count)

                    extra_pair = []
                    for two in two_list:
                        if two == pair:
                            continue
                        two_val = MoveGenerator.get_value(two)
                        if two_val == 1:
                            if new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 2:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(
                                    two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 8:
                            if new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(
                                    two - 1) or new_cards_to_count_copy.get(two - 2):
                                extra_pair.append([two] * 2)
                        elif two_val == 9:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(two - 2):
                                extra_pair.append([two] * 2)
                        else:
                            if new_cards_to_count_copy.get(two - 1) or new_cards_to_count_copy.get(two - 2) or \
                                    new_cards_to_count_copy.get(two + 1) or new_cards_to_count_copy.get(two + 2):
                                extra_pair.append([two] * 2)

                    all_split_cards.extend(extra_pair)
                    ke_zi = [[card] * 3 for card in three_list + four_list if card != pair]  # 刻子也添加进去
                    all_split_cards.extend(ke_zi)
                    all_split_cards.extend(extra_comb)

                    # 先拆顺子
                    # TODO: 再计算搭子(二连顺、间隔顺)
                    curr_heap_idx = 0  # 当前堆的索引
                    record_comb = ...
                    all_comb = itertools.combinations(range(len(all_split_cards)), need_heap)
                    # all_comb_list = list(all_comb)
                    # print("所有组合长度：", len(all_comb_list), all_comb_list)
                    for comb in all_comb:
                        if comb[:curr_heap_idx + 1] == record_comb:
                            continue
                        # 统计每一张手牌数量
                        cards_to_count_copy = new_cards_to_count.copy()
                        comb_list = []
                        curr_heap_idx = 0
                        record_comb = ...
                        flag = True
                        # 根据所需的搭子数，计算搭子
                        for i in range(need_heap):
                            curr_heap_idx = i
                            one_comb = all_split_cards[comb[i]]
                            for c in one_comb:
                                if cards_to_count_copy.get(c) <= 0:
                                    flag = False
                                    record_comb = comb[:i + 1]
                                    break
                                cards_to_count_copy[c] -= 1
                            if not flag:
                                comb_list.clear()
                                break
                            comb_list.append(one_comb)

                        if comb_list:
                            res_split_cards = []
                            res_split_cards.extend(split_path)
                            res_split_cards.extend(comb_list)

                            # 平胡拆牌后，计算向听数 注意：res_split_cards就是根据需要堆数的组合，所以不用考虑多个对子的向听数问题
                            xts = the_worst_xts
                            for sc in res_split_cards:
                                comb_type = MoveGenerator.judge_cards_type(sc)
                                comb_xts = COMB_REMOVE_XTS.get(comb_type, 0)
                                xts -= comb_xts

                            for card, count in cards_to_count_copy.items():
                                count > 0 and res_split_cards.append([card] * count)

                            if xts < record_lowest_xts:
                                optimal_path.clear()
                                optimal_path.append(res_split_cards)
                                record_lowest_xts = xts
                            elif xts == record_lowest_xts:
                                optimal_path.append(res_split_cards)

            optimal_split_cards(new_hand_cards)

        ting_cards = self.calc_ping_hu_by_ting_cards(optimal_path)

        return record_lowest_xts, ting_cards

    def calc_ping_hu_by_ting_cards(self, optimal_paths):
        """
        计算一轮结束后，真人玩家叫牌后所听的牌
        """
        ting_cards = []
        for idx, optimal_path in enumerate(optimal_paths):
            for cards in optimal_path:
                if len(cards) == 1:
                    ting_cards.extend(cards)
                if len(cards) == 2:
                    result = self.calc_two_cards_by_yxp(cards)
                    ting_cards.extend(result)

        return list(set(ting_cards))

    def calc_two_cards_by_yxp(self, cards):
        """
        计算卡牌为两张时，有效听牌
        """
        ting_yxp = []
        if cards[0] == cards[-1]:
            return []

        elif cards[0]+1 == cards[-1]:
            if 0 < self.get_value(cards[0]) - 1 <= 9:
                ting_yxp.append(cards[0] - 1)

            if 0 < self.get_value(cards[-1]) + 1 <= 9:
                ting_yxp.append(cards[-1] + 1)

        elif cards[-1] - cards[0] == 2:
            if 0 < self.get_value(cards[0]) + 1 <= 9:
                ting_yxp.append(cards[0] + 1)

        return ting_yxp

    @staticmethod
    def unpack_deduplicate_cards(deduplicate, ting_cards):
        """
        解包最优卡牌
        """
        one_cards, two_cards, three_cards = [], [], []
        for unpack_cards in deduplicate:
            for cards in unpack_cards:
                if len(cards) == 1:
                    one_cards.extend(cards)
                elif len(cards) == 2:
                    two_cards.extend(cards)
                elif len(cards) == 3:
                    three_cards.extend(cards)
        one_cards = list(set(one_cards) - set(ting_cards))
        two_cards = list(set(two_cards) - set(ting_cards))
        three_cards = list(set(three_cards) - set(ting_cards))

        return one_cards, two_cards, three_cards

    @staticmethod
    def calc_gap_cards(cards1, cards2):
        """
        计算是否为间隔的两张卡牌
        """
        # 判断两张卡牌是否为间隔牌
        if (cards1 % 10) == (cards2 % 10):
            if abs((cards1 // 10) - (cards2 // 10)) == 2:
                return True

        return False

    def cal_xts_da_dui_zi(self, *args):
        """ 大对子 """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.DA_DUI_ZI)
        one_list, two_list, three_list, four_list = args
        need_heap = self.__hand_cards_len // 3 + 1  # 刻子数 + 一对
        two_len = len(two_list)
        real_xts = the_worst_xts - two_len if two_len <= need_heap else the_worst_xts - need_heap
        real_xts -= len(three_list + four_list) * 2

        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 3)
            extra_one.append(four)
        for three in three_list:
            optimal_split_cards.append([three] * 3)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        yxp_cards = set()
        two_list_len = len(two_list)
        for path in optimal_split_cards:
            if len(path) == 3:
                continue
            if two_list_len != 1:
                yxp_cards.add(path[0])

        extra_one.extend(one_list)
        record_chu_pai = {}
        if extra_one:
            optimal_split_cards.extend([[one] for one in extra_one])
            for c in extra_one:
                yxp_copy = yxp_cards.copy()
                yxp_copy.update(set(extra_one))
                yxp_copy.remove(c)
                record_chu_pai[c] = yxp_copy
        else:
            if len(two_list) > 1:
                for c in yxp_cards:
                    yxp_copy = yxp_cards.copy()
                    yxp_copy.remove(c)
                    record_chu_pai[c] = yxp_copy

        best_cards = self.get_best_card(record_chu_pai)
        print(f"大对子最优组合：{optimal_split_cards}")

        return real_xts, best_cards

    def cal_xts_da_dui_zi_no_dian_pao(self, ting_cards, *args):
        """
        TODO: 计算大对子出牌不被其他玩家胡牌
        胡大牌: 小七对、大对子、龙七对、清一色...
        """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.DA_DUI_ZI)
        one_list, two_list, three_list, four_list = args
        need_heap = self.__hand_cards_len // 3 + 1  # 刻子数 + 一对
        two_len = len(two_list)
        real_xts = the_worst_xts - two_len if two_len <= need_heap else the_worst_xts - need_heap
        real_xts -= len(three_list + four_list) * 2

        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 3)
            extra_one.append(four)
        for three in three_list:
            optimal_split_cards.append([three] * 3)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        yxp_cards = set()
        two_list_len = len(two_list)
        for path in optimal_split_cards:
            if len(path) == 3:
                continue
            if two_list_len != 1:
                yxp_cards.add(path[0])

        extra_one.extend(one_list)
        record_chu_pai = {}
        if extra_one:
            optimal_split_cards.extend([[one] for one in extra_one])
            for c in extra_one:
                yxp_copy = yxp_cards.copy()
                yxp_copy.update(set(extra_one))
                yxp_copy.remove(c)
                record_chu_pai[c] = yxp_copy
        else:
            if len(two_list) > 1:
                for c in yxp_cards:
                    yxp_copy = yxp_cards.copy()
                    yxp_copy.remove(c)
                    record_chu_pai[c] = yxp_copy

        # TODO: 添加出牌风险逻辑处理(点炮被胡大牌)
        # 避免大对子有效出牌存在被大牌点炮胡风险
        # new_record_chu_pai = {**record_chu_pai}
        # 当有效出牌存在听牌列表中时，将其删除
        for card in list(record_chu_pai.keys()):
            if card in ting_cards:
                record_chu_pai.pop(card, None)
            continue

        # 判断是否所有有效牌是否都不能打出
        if not record_chu_pai:
            # 当单牌和对子不能选择时，从刻子或杠牌(憨包杠)中选择
            one_cards = list(set(one_list) - set(ting_cards))
            if one_cards:
                print(f"大对子最优组合：{optimal_split_cards}")
                return real_xts, one_cards

            two_cards = list(set(two_list) - set(ting_cards))
            if two_cards:
                print(f"大对子最优组合：{optimal_split_cards}")
                return real_xts + 1, two_cards

            best_cards = list(set(three_list + four_list) - set(ting_cards))
            if best_cards:
                print(f"大对子最优组合：{optimal_split_cards}")
                return real_xts + 1, best_cards

        best_cards = self.get_best_card(record_chu_pai)
        # 无最优出牌，出那一张都被必胡，则从手牌随机选择可出卡牌
        if not best_cards:
            best_cards = self.__hand_cards[:]
        print(f"大对子最优组合：{optimal_split_cards}")

        return real_xts, best_cards

    def get_round_over_by_ddz_ting_cards(self, *args):
        """
        计算一轮结束后，真人玩家叫牌类型为大队子所听卡牌
        """
        the_worst_xts = self.round_over_the_worst_xts_by_hu_type.get(HuPaiType.DA_DUI_ZI)
        one_list, two_list, three_list, four_list = args
        need_heap = self.__hand_cards_len // 3 + 1  # 刻子数 + 一对
        two_len = len(two_list)
        real_xts = the_worst_xts - two_len if two_len <= need_heap else the_worst_xts - need_heap
        real_xts -= len(three_list + four_list) * 2

        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 3)
            extra_one.append(four)
        for three in three_list:
            optimal_split_cards.append([three] * 3)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        yxp_cards = set()
        two_list_len = len(two_list)
        for path in optimal_split_cards:
            if len(path) == 3:
                continue
            if two_list_len != 1:
                yxp_cards.add(path[0])

        extra_one.extend(one_list)
        record_chu_pai = {}
        if extra_one:
            optimal_split_cards.extend([[one] for one in extra_one])
            for c in extra_one:
                yxp_copy = yxp_cards.copy()
                yxp_copy.update(set(extra_one))
                yxp_copy.remove(c)
                record_chu_pai[c] = yxp_copy
        else:
            if len(two_list) > 1:
                for c in yxp_cards:
                    yxp_copy = yxp_cards.copy()
                    yxp_copy.remove(c)
                    record_chu_pai[c] = yxp_copy

        return real_xts, list(record_chu_pai.keys())

    def cal_xts_by_di_long_qi(self, *args):
        """ 计算当前手牌按地龙七胡牌类型的向听数 """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.DI_LONG_QI)  # 最坏向听数
        one_list, two_list, three_list, four_list = args
        real_xts = the_worst_xts - len(two_list)
        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 4)
        for three in three_list:
            optimal_split_cards.append([three] * 2)
            extra_one.append(three)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        extra_one.extend(one_list)  # 单牌
        optimal_split_cards.extend([[one] for one in extra_one])
        record_chu_pai = {}
        for c in extra_one:
            yxp_cards = set(extra_one)
            yxp_cards.remove(c)
            record_chu_pai[c] = yxp_cards

        best_cards = self.get_best_card(record_chu_pai)
        print(f"地龙七最优组合：{optimal_split_cards}, 地龙七最终打牌: {best_cards}")

        return real_xts, best_cards

    def cal_xts_by_di_long_qi_no_dian_pao(self, ting_cards, *args):
        """
        TODO: 计算当前手牌按地龙七胡牌类型的向听数及出牌不被胡大牌
        胡大牌: 小七对、大对子、龙七对、清一色...
        """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.DI_LONG_QI)  # 最坏向听数
        one_list, two_list, three_list, four_list = args
        real_xts = the_worst_xts - len(two_list)
        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 4)
        for three in three_list:
            optimal_split_cards.append([three] * 2)
            extra_one.append(three)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        extra_one.extend(one_list)  # 单牌
        optimal_split_cards.extend([[one] for one in extra_one])
        record_chu_pai = {}
        for c in extra_one:
            yxp_cards = set(extra_one)
            yxp_cards.remove(c)
            record_chu_pai[c] = yxp_cards

        # TODO: 添加出牌风险逻辑处理(点炮被胡大牌)
        # 避免大对子有效出牌存在被大牌点炮胡风险
        new_record_chu_pai = {**record_chu_pai}
        # 当有效出牌存在听牌列表中时，将其删除
        for card in list(new_record_chu_pai.keys()):
            if card in ting_cards:
                new_record_chu_pai.pop(card, None)
            continue

        # 判断是否所有有效牌是否都不能打出
        if not new_record_chu_pai:
            # 当单牌和刻字不能选择时，从对子或杠牌(憨包杠)中选择
            best_cards = two_list + four_list
            print(f"地七对最优组合：{optimal_split_cards}")
            return real_xts + 1, best_cards

        best_cards = self.get_best_card(record_chu_pai)
        print(f"地龙七最优组合：{optimal_split_cards}, 地龙七最终打牌: {best_cards}")

        return real_xts, best_cards

    def get_round_over_by_dlq_ting_cards(self, *args):
        """
        计算一轮结束后，真人玩家叫牌类型为地龙七所听卡牌
        """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.DI_LONG_QI)  # 最坏向听数
        one_list, two_list, three_list, four_list = args
        real_xts = the_worst_xts - len(two_list)
        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 4)
        for three in three_list:
            optimal_split_cards.append([three] * 2)
            extra_one.append(three)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        extra_one.extend(one_list)  # 单牌
        optimal_split_cards.extend([[one] for one in extra_one])
        record_chu_pai = {}
        for c in extra_one:
            yxp_cards = set(extra_one)
            yxp_cards.remove(c)
            record_chu_pai[c] = yxp_cards

        return real_xts, list(record_chu_pai.keys())

    def cal_xts_by_qi_dui(self, *args):
        """ 计算当前手牌按七对胡牌类型的向听数 """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.QI_DUI)  # 最坏向听数
        one_list, two_list, three_list, four_list = args  # self.calc_cards_list_by_count()
        reduce_one_xts = len(two_list + three_list)
        reduce_two_xts = len(four_list) * 2
        real_xts = the_worst_xts - reduce_one_xts - reduce_two_xts  # 真实向听数
        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 4)
        for three in three_list:
            optimal_split_cards.append([three] * 2)
            extra_one.append(three)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        extra_one.extend(one_list)  # 单牌
        optimal_split_cards.extend([[one] for one in extra_one])
        record_chu_pai = {}
        for c in extra_one:
            yxp_cards = set(extra_one)
            yxp_cards.remove(c)
            record_chu_pai[c] = yxp_cards

        best_cards = self.get_best_card(record_chu_pai)
        print(f"七对最优组合：{optimal_split_cards}")

        return real_xts, best_cards

    def cal_xts_by_qi_dui_no_dian_pao(self, ting_cards, *args):
        """
        TODO: 计算当前手牌按七对胡牌类型的向听数及出牌不被胡大牌
        胡大牌: 小七对、大对子、龙七对、清一色...
        """
        the_worst_xts = self.the_worst_xts_by_hu_type.get(HuPaiType.QI_DUI)  # 最坏向听数
        one_list, two_list, three_list, four_list = args  # self.calc_cards_list_by_count()
        reduce_one_xts = len(two_list + three_list)
        reduce_two_xts = len(four_list) * 2
        real_xts = the_worst_xts - reduce_one_xts - reduce_two_xts  # 真实向听数
        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 4)
        for three in three_list:
            optimal_split_cards.append([three] * 2)
            extra_one.append(three)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        extra_one.extend(one_list)  # 单牌
        optimal_split_cards.extend([[one] for one in extra_one])
        record_chu_pai = {}
        for c in extra_one:
            yxp_cards = set(extra_one)
            yxp_cards.remove(c)
            record_chu_pai[c] = yxp_cards

        # TODO: 添加出牌风险逻辑处理(点炮被胡大牌)
        # 避免大对子有效出牌存在被大牌点炮胡风险
        # new_record_chu_pai = {**record_chu_pai}
        # 当有效出牌存在听牌列表中时，将其删除
        for card in list(record_chu_pai.keys()):
            if card in ting_cards:
                record_chu_pai.pop(card, None)
            continue

        # 判断是否所有有效牌是否都不能打出
        if not record_chu_pai:
            # 当单牌和刻字不能选择时，从对子或杠牌(憨包杠)中选择
            best_cards = list(set(two_list + four_list) - set(ting_cards))
            print(f"小七对最优组合：{optimal_split_cards}")
            return real_xts + 1, best_cards

        best_cards = self.get_best_card(record_chu_pai)
        # 无最优出牌，出那一张都被必胡，则从手牌随机选择可出卡牌
        if not best_cards:
            return real_xts, self.__hand_cards
        print(f"七对最优组合：{optimal_split_cards}")

        return real_xts, best_cards

    def get_round_over_by_xqd_ting_cards(self, *args):
        """
        计算一轮结束后，真人玩家叫牌类型为小七对所听卡牌
        """
        the_worst_xts = self.round_over_the_worst_xts_by_hu_type.get(HuPaiType.QI_DUI)  # 最坏向听数
        one_list, two_list, three_list, four_list = args  # self.calc_cards_list_by_count()
        reduce_one_xts = len(two_list + three_list)
        reduce_two_xts = len(four_list) * 2
        real_xts = the_worst_xts - reduce_one_xts - reduce_two_xts  # 真实向听数
        extra_one = []
        optimal_split_cards = []
        for four in four_list:
            optimal_split_cards.append([four] * 4)
        for three in three_list:
            optimal_split_cards.append([three] * 2)
            extra_one.append(three)
        for two in two_list:
            optimal_split_cards.append([two] * 2)

        extra_one.extend(one_list)  # 单牌
        optimal_split_cards.extend([[one] for one in extra_one])
        record_chu_pai = {}
        for c in extra_one:
            yxp_cards = set(extra_one)
            yxp_cards.remove(c)
            record_chu_pai[c] = yxp_cards

        return real_xts, list(record_chu_pai.keys())

    def cal_xts_long_qi_dui(self, *args):
        """ 龙七对 """
        return self.cal_xts_by_qi_dui(*args)

    def cal_xts_long_qi_dui_no_dian_pao(self, *args):
        """
        TODO: 计算当前手牌按龙七对胡牌类型的向听数及出牌不被胡大牌
        胡大牌: 小七对、大对子、龙七对、清一色...
        """
        return self.cal_xts_by_qi_dui_no_dian_pao(*args)

    def calc_qing_yi_se(self):
        """
        检查手牌是否都是同一颜色
        """
        tmp_hand_cards = self.__hand_cards[:]
        cards_by_type = defaultdict(list)
        # 计算卡牌花色和牌值
        for card in tmp_hand_cards:
            # 筛选掉万能牌，万能牌可作为清一色牌型
            if card == CardType.LAI_ZI:
                continue
            card_type = card // 10
            cards_by_type[card_type].append(card)
        # 返回卡牌类型和牌值
        return cards_by_type

    def cal_yxp_by_ping_hu(self, optimal_path, yx_heap, xts):
        """
        根据胡牌类型计算有效牌
        1.平胡
        2.小七对
        3.大队子
        4.龙七对
        [11, 12, 15, 16] -> 有效牌组: [10, 13, 14, 17]
        :params optimal_path 最优路径 -> [[], [], []]
        :params the_worst_xts 仅根据牌数计算出的当前牌的向听数
        :params xts 向听数
        :params yx_heap 有效堆（需要堆数）
        """
        if xts < 0:
            # todo: 后续看需求是否增加改牌
            print("开始检测手牌是否形成胡牌，重新选择最优出牌!")
            count = 0
            best_cards = []
            remain_cards_by_deck = self.calc_remain_cards_by_deck()
            for card in self.__hand_cards:
                tmp_count = 0
                tmp_count += remain_cards_by_deck[card]
                if tmp_count > count:
                    best_cards.clear()
                    count = tmp_count
                    best_cards.append(card)

            # 找不到条件符合的卡牌时，从当前手牌中，随机选择一张
            if not best_cards:
                return self.__hand_cards

            return list(set(best_cards))

        # 胡牌类型组的有效卡牌
        record_chu_pai = {}
        for i, path in enumerate(optimal_path):
            yxp_cards = set()
            # 先算算有效堆的有效牌
            for j, cards in enumerate(path[:yx_heap + 1]):
                # 不计算面子有效牌
                cards_len = len(cards)
                if cards_len == 3:
                    continue
                # 计算单张卡牌有效牌(i-1, i, i+1)
                if cards_len == 1:
                    card = cards[0]
                    yxp_cards.add(card)
                    card_val = MoveGenerator.get_value(card)
                    if xts != 0:
                        # 计算有效牌(先处理边界，再处理中间)
                        if card_val == 1:
                            yxp_cards.add(card + 1)
                            yxp_cards.add(card + 2)
                        elif card_val == 9:
                            yxp_cards.add(card - 1)
                            yxp_cards.add(card - 2)  # 坎张
                        else:
                            yxp_cards.add(card - 1)
                            if card_val > 2:
                                yxp_cards.add(card - 2)
                            yxp_cards.add(card + 1)
                            if card_val < 8:
                                yxp_cards.add(card + 2)
                elif cards_len == 2:
                    card1 = cards[0]
                    card2 = cards[1]
                    # 计算将牌情况(将牌对数大于1时，再计算)
                    if (j != 0 or xts > 0) and card1 == card2:
                        yxp_cards.add(card1)
                    # 计算二连顺(只处理边界)
                    elif card2 - card1 == 1:
                        card1_val = MoveGenerator.get_value(card1)
                        card2_val = MoveGenerator.get_value(card2)
                        if card1_val != 1:
                            yxp_cards.add(card1 - 1)
                        if card2_val != 9:
                            yxp_cards.add(card2 + 1)
                    # 计算间隔顺(只处理中间)
                    elif card2 - card1 == 2:
                        yxp_cards.add(card2 - 1)

            # 在算无效堆的有效牌
            invalid_heap_cards = []
            for cards in path[yx_heap + 1:]:
                invalid_heap_cards.extend(cards)
            invalid_heap_cards = list(set(invalid_heap_cards))
            for card in invalid_heap_cards:
                record_chu_pai.setdefault(card, set()).update(yxp_cards)

        return self.get_best_card(record_chu_pai, ping_hu=True)

    def get_best_card(self, record_chu_pai: dict, ping_hu=False):
        yxp_count = 0
        best_cards = []
        for card, yxp in record_chu_pai.items():
            count = 0
            for c in yxp:
                count += self.res_cards_to_count.get(c, 0)
            if count > yxp_count:
                best_cards.clear()
                yxp_count = count
                best_cards.append(card)
            elif count == yxp_count:
                best_cards.append(card)
        if ping_hu:
            final_cards = []
            for c in best_cards:
                c_v = self.get_value(c)
                if c_v == 1 or c_v == 9:  # 有限打边张
                    final_cards.append(c)
            return final_cards or best_cards
        return best_cards

    def cal_yxp_by_ping_hu_no_dian_pao(self, optimal_path, yx_heap, xts, ting_cards):
        """
        TODO: 当前出牌不会被大牌胡牌
        根据胡牌类型计算有效牌
        1.平胡
        2.小七对
        3.大队子
        4.龙七对
        [11, 12, 15, 16] -> 有效牌组: [10, 13, 14, 17]
        :params optimal_path 最优路径 -> [[], [], []]
        :params the_worst_xts 仅根据牌数计算出的当前牌的向听数
        :params xts 向听数
        :params yx_heap 有效堆（需要堆数）
        """
        if xts < 0:
            # todo: 后续看需求是否增加改牌
            print("开始检测手牌是否形成胡牌，拆解手牌选择最优出牌!")
            count = 0
            best_cards = []
            remain_cards_by_deck = self.calc_remain_cards_by_deck()
            for card in self.__hand_cards:
                tmp_count = 0
                tmp_count += remain_cards_by_deck[card]
                if tmp_count > count:
                    best_cards.clear()
                    count = tmp_count
                    best_cards.append(card)

            # 找不到条件符合的卡牌时，从当前手牌中，随机选择一张
            if not best_cards:
                return self.__hand_cards

            return list(set(best_cards))

        # 胡牌类型组的有效卡牌
        record_chu_pai = {}
        for i, path in enumerate(optimal_path):
            yxp_cards = set()
            # 先算算有效堆的有效牌
            for j, cards in enumerate(path[:yx_heap + 1]):
                # 不计算面子有效牌
                cards_len = len(cards)
                if cards_len == 3:
                    continue
                # 计算单张卡牌有效牌(i-1, i, i+1)
                if cards_len == 1:
                    card = cards[0]
                    yxp_cards.add(card)
                    card_val = MoveGenerator.get_value(card)
                    if xts != 0:
                        # 计算有效牌(先处理边界，再处理中间)
                        if card_val == 1:
                            yxp_cards.add(card + 1)
                            yxp_cards.add(card + 2)
                        elif card_val == 9:
                            yxp_cards.add(card - 1)
                            yxp_cards.add(card - 2)  # 坎张
                        else:
                            yxp_cards.add(card - 1)
                            if card_val > 2:
                                yxp_cards.add(card - 2)
                            yxp_cards.add(card + 1)
                            if card_val < 8:
                                yxp_cards.add(card + 2)
                elif cards_len == 2:
                    card1 = cards[0]
                    card2 = cards[1]
                    # 计算将牌情况(将牌对数大于1时，再计算)
                    if (j != 0 or xts > 0) and card1 == card2:
                        yxp_cards.add(card1)
                    # 计算二连顺(只处理边界)
                    elif card2 - card1 == 1:
                        card1_val = MoveGenerator.get_value(card1)
                        card2_val = MoveGenerator.get_value(card2)
                        if card1_val != 1:
                            yxp_cards.add(card1 - 1)
                        if card2_val != 9:
                            yxp_cards.add(card2 + 1)
                    # 计算间隔顺(只处理中间)
                    elif card2 - card1 == 2:
                        yxp_cards.add(card2 - 1)

            # 在算无效堆的有效牌
            invalid_heap_cards = []
            for cards in path[yx_heap + 1:]:
                invalid_heap_cards.extend(cards)
            invalid_heap_cards = list(set(invalid_heap_cards))
            for card in invalid_heap_cards:
                record_chu_pai.setdefault(card, set()).update(yxp_cards)

        # TODO: 添加出牌风险逻辑处理(点炮被胡大牌)
        # 避免大对子有效出牌存在被大牌点炮胡风险
        for card in list(record_chu_pai.keys()):
            # 当有效出牌存在听牌列表中时，将其删除，在计算最优出牌
            if card in ting_cards:
                record_chu_pai.pop(card, None)
            continue

        return self.get_best_card(record_chu_pai, ping_hu=True)

    def calc_remain_cards_by_deck(self):
        """
        计算牌堆里剩余的卡牌
        """
        # 统计卡牌数量
        remain_cards_dict = self.calc_cards_to_count(self.others_hand_cards)
        remain_cards_by_deck = {int(key): value for key, value in self.res_cards_to_count.items()}
        for card, nums in remain_cards_dict.items():
            if remain_cards_by_deck.get(card, 0):
                remain_cards_by_deck[card] -= nums

        return remain_cards_by_deck

    def get_round_over_call_type_and_ting_cards(self, ting_list, others_cards_and_piles):
        """
        TODO: 计算一轮结束后，真人玩家叫牌类型和听牌
        真人玩家向听数为0 -> 叫牌
        真人玩家向听数不为0 -> 未叫牌，不计算
        """
        # 一轮结束后，真人玩家听牌
        round_over_by_ting_cards = []
        # 添加已经锁牌之后的听牌
        if ting_list:
            round_over_by_ting_cards.extend(ting_list)

        # 计算真人玩家叫牌，且不为平胡时所听大牌
        for cards in others_cards_and_piles:
            hand_cards, piles = cards[0], cards[1]
            args = self.calc_round_over_cards_list_by_count(hand_cards)

            # 更新一轮结束后，其他玩家牌型向听数(更新当前牌型向听数)
            self.get_round_over_by_update_attr(hand_cards)

            # 平胡仅判断清一色即可，正常普通平胡不需要判断
            if self.get_round_over_by_qys(hand_cards, piles):
                ph_xts, ph_ting_cards = self.get_round_over_by_ping_hu_ting_cards(hand_cards, *args)
                if ph_xts == 0:
                    round_over_by_ting_cards.extend(ph_ting_cards)

            # 大队子
            ddz_xts, ddz_ting_cards = self.get_round_over_by_ddz_ting_cards(*args)
            if ddz_xts == 0:
                round_over_by_ting_cards.extend(ddz_ting_cards)

            # 小七对
            if len(hand_cards) == 13:
                xqd_xts, xqd_ting_cards = self.get_round_over_by_xqd_ting_cards(*args)
                if xqd_xts == 0:
                    round_over_by_ting_cards.extend(xqd_ting_cards)

        if round_over_by_ting_cards:
            print("输出一轮结束后，真人玩家所听大牌: {}".format(list(set(round_over_by_ting_cards))))

        return list(set(round_over_by_ting_cards))

    def get_round_over_by_qys(self, hand_cards, piles):
        """
        判断真人玩家手牌或与碰、杠是否为清一色
        """
        pg_cards = sum([pile[1:-1] for pile in piles], [])
        pg_cards_type = self.calc_same_card_type(pg_cards)
        hd_cards_type = self.calc_same_card_type(hand_cards)
        pg_types = list(pg_cards_type.keys())
        hd_types = list(hd_cards_type.keys())
        # 判断真人玩家未碰、杠时，听大牌是否为清一色
        if len(hd_types) == 1:
            return True

        # 判断真人玩家碰、杠后，听大牌是否清一色
        if len(pg_types) == 1 and len(hd_types) == 1:
            if pg_types[0] == hd_types[0]:
                return True

        return False

    def calc_round_over_cards_list_by_count(self, cards):
        """
        计算叫牌玩家手牌情况
        """
        # one_list, two_list, three_list, four_list = [], [], [], []
        one_list = []
        two_list = []
        three_list = []
        four_list = []
        cards_dict = self.calc_cards_to_count(cards)
        for card, count in cards_dict.items():
            if count == 1:
                one_list.append(card)
            elif count == 2:
                two_list.append(card)
            elif count == 3:
                three_list.append(card)
            else:
                four_list.append(card)
        return one_list, two_list, three_list, four_list

    def calc_cards_to_count(self, cards=None) -> dict:
        """
        计算出手牌每张牌的数量
        已经去除癞子
        cards: 如果不传则就计算手牌
        """
        if not cards:
            if not self.__cards_to_count:
                cards = self.__hand_cards[:]
                self.remove_by_value(cards, CardType.LAI_ZI, -1)
                self.__cards_to_count = self.cards_to_count(cards)
            return self.__cards_to_count
        else:
            self.remove_by_value(cards, CardType.LAI_ZI, -1)
            return self.cards_to_count(cards)

    def calc_cards_list_by_count(self):
        """ 计算不同数量的cards """
        one_list = []  # 单张卡牌
        two_list = []  # 对子
        three_list = []  # 刻子
        four_list = []  # 四张
        cards_to_count = self.calc_cards_to_count()
        for card, count in cards_to_count.items():
            if count == 1:
                one_list.append(card)
            elif count == 2:
                two_list.append(card)
            elif count == 3:
                three_list.append(card)
            else:
                four_list.append(card)

        return one_list, two_list, three_list, four_list

    @staticmethod
    def remove_by_value(data, value, remove_count=1):
        """
        删除列表data中的value
        :param data: list
        :param value:
        :param remove_count: 为-1的时候表示删除全部, 默认为1
        :return: already_remove_count: int
        """
        data_len = len(data)
        count = remove_count == -1 and data_len or remove_count

        already_remove_count = 0

        for i in range(0, count):
            if value in data:
                data.remove(value)
                already_remove_count += 1
            else:
                break

        return already_remove_count

    @staticmethod
    def gen_serial_moves(single_cards, min_serial=2, max_serial=3, repeat=1, solid_num=0):
        """
        减去 对子/刻子 后的所有搭子（两面，坎张，顺子）
        params: cards 输入牌
        params: min_serial  最小连数
        params: max_serial  最大连数
        params: repeat  重复牌数
        params: solid_num  固定连数
        拆牌：坎张|两面|
        """
        seq_records = list()
        result = list()

        cards_len = len(single_cards)

        # 至少重复数是最小序列
        if solid_num < min_serial:
            solid_num = 0

        # 顺子（最少2张）
        start = i = 0
        longest = 1
        while i < cards_len:
            # 判断连续两张牌
            if i + 1 < cards_len and single_cards[i + 1] - single_cards[i] == 1:
                longest += 1
                i += 1
            else:
                # 记录索引
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        for seq in seq_records:
            if seq[1] < min_serial:
                continue
            start, longest = seq[0], seq[1]
            longest_list = single_cards[start: start + longest]

            if solid_num == 0:  # No limitation on how many sequences
                steps = min_serial  # 最小连数
                while steps <= longest:
                    index = 0
                    while steps + index <= longest:
                        target_moves = sorted(longest_list[index: index + steps] * repeat)
                        result.append(target_moves)
                        index += 1
                    steps += 1  # 递增
                    if steps > max_serial:
                        break
            else:
                if longest < solid_num:
                    continue
                index = 0
                while index + solid_num <= longest:
                    target_moves = sorted(longest_list[index: index + solid_num] * repeat)
                    result.append(target_moves)
                    index += 1
        # 坎张
        i = 0
        while i < cards_len:
            # 连续的两张坎张
            start_val = single_cards[i] % 10
            if start_val > 7:
                i += 1
                continue
            if i + 1 < cards_len and single_cards[i] + 2 == single_cards[i + 1]:
                result.append(single_cards[i: i + min_serial])
            # 间隔的两张坎张
            if i + 2 < cards_len and single_cards[i] + 2 == single_cards[i + 2]:
                result.append([single_cards[i], single_cards[i + 2]])
            i += 1

        return result

    @staticmethod
    def get_repeat_cards(cards):
        """
        params: cards 输入牌
        params: min_num  最小连数
        params: max_serial  最大连数
        params: solid_num  固定连数
        找出相同的牌，如：22,22 | 22,22,22
        找出坎张，如：11,13| 22,24
        """
        result = []
        card_to_count = MoveGenerator.cards_to_count(cards)

        for card, count in card_to_count.items():
            if count <= 1:
                continue
            if count == 4:
                result.append([card] * 2)
                result.append([card] * 3)
                result.append([card] * 4)
            elif count == 3:
                result.append([card] * 2)
                result.append([card] * 3)
            else:
                result.append([card] * 2)
        return result

    @staticmethod
    def cards_to_count(cards):
        """
        统计卡牌数量(dict)
        """
        card_to_count = dict()
        for c in cards:
            card_to_count[c] = card_to_count.get(c, 0) + 1
        return card_to_count

    @staticmethod
    def get_value(card):
        return card % 10

def cal_time(func, *args, **kwargs):
    def inner():
        import time
        s = time.time()
        func(*args, **kwargs)
        print("{}耗时：".format(func.__name__), time.time() - s)

    return inner

def split_test():
    """
    测试拆牌
    """
    mg = MoveGenerator()
    test_cards = [11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 19, 22]  # 每一张都胡
    res1 = mg.get_repeat_cards(test_cards)
    res2 = mg.gen_serial_moves(test_cards)
    print(res1, len(res1))
    print(res2, len(res2))
    print()