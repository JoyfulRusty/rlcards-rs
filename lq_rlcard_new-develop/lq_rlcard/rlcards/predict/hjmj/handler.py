# -*- coding: utf-8 -*-

from collections import Counter
from base.base_server import BaseServer
from rlcards.games.mahjong.xts import MoveGenerator
from rlcards.games.mahjong.hjmj_xts import HJMoveGenerator
from rlcards.games.mahjong.yxp import calc_best_cards_by_hfc, calc_best_cards_by_xxc

class MahjongHandler(BaseServer):
    """
    麻将处理函数
    """
    def __init__(self, server_name):
        super().__init__(server_name)

    async def cal_action(self, data):
        """
        计算出牌动作
        """
        await self.info_log(data)  # 日志记录

        print("从队列中读取数据: {}".format(data))
        print("######-> AI开始计算预测动作 <-######")

        if not data.get("xxc_level") or 0:
            action = self.calc_hfc_actions(data)
        else:
            action = self.calc_xxc_actions(data)

        data['cards'] = action

        await self.info_log(f"计算出牌, tid: {data.get('tid')}, uid: {data.get('uid')}, action: {action}")
        print('当前预测玩家ID为: {}'.format(data['self']))
        print('AI输出预测打牌动作: {}'.format(action))
        print()
        print("%##############<< 预测下一位玩家出牌动作 >>##################%")
        await self.send_child_game(data, action)

    async def cal_pong(self, data):
        """
        计算是否碰
        """
        await self.info_log(data)  # 日志记录

        print("从队列中读取数据: {}".format(data))
        print("######-> AI开始计算预测动作 <-######")

        mg = MoveGenerator()
        # 计算剩余卡牌并更新卡牌属性
        remain_cards = self.calc_remain_cards(data.get("curr_hand_cards") or [], data.get("remain_cards") or {})
        mg.update_attr(
            data.get("curr_hand_cards"),
            data.get("piles"),
            data.get("left_count"),
            data.get("others_hand_cards"),
            remain_cards
        )

        # 判断小七对是否进行碰操作
        action = mg.calc_can_xqd_pong(data.get("curr_card"))

        data['cards'] = action

        await self.info_log(f"计算碰, tid: {data.get('tid')}, uid: {data.get('uid')}, action: {action}")
        print('当前预测玩家ID为: {}'.format(data['self']))
        print('AI输出预测碰牌动作: {}'.format(action))
        print()
        print("%##############<< 预测下一位玩家出牌动作 >>##################%")
        await self.send_child_game(data, action)

    async def cal_gang(self, data):
        """
        计算是否杠
        """
        await self.info_log(data)  # 日志记录

        print("从队列中读取数据: {}".format(data))
        print("######-> AI开始计算预测动作 <-######")

        mg = MoveGenerator()
        # 计算剩余卡牌并更新卡牌属性
        remain_cards = self.calc_remain_cards(data.get("curr_hand_cards") or [], data.get("remain_cards") or {})
        mg.update_attr(
            data.get("curr_hand_cards"),
            data.get("piles"),
            data.get("left_count"),
            data.get("others_hand_cards"),
            remain_cards
        )

        # 计算玩家杠动作
        print("当前杠牌操作类型: {}".format(data.get("gang_type")))

        action = mg.calc_can_gang(data.get("can_gang_cards"), data.get("gang_type"))

        data['cards'] = action

        await self.info_log(f"计算杠, tid: {data.get('tid')}, uid: {data.get('uid')}, action: {action}")
        print('当前预测玩家ID为: {}'.format(data['self']))
        print('AI输出预测杠牌动作: {}'.format(action))
        print()
        print("%##############<< 预测下一位玩家出牌动作 >>##################%")
        await self.send_child_game(data, action)

    async def cal_yxp(self, data):
        """
        计算机器人摸好牌
        """
        await self.info_log(data)  # 日志记录

        print("从队列中读取数据: {}".format(data))
        print("######-> AI开始计算预测最佳有效牌 <-######")

        # 选择话费场或休闲场摸好牌
        if not data.get("xxc_level") or 0:
            print("============话费场摸好牌计算============")
            action, hu_cards = calc_best_cards_by_hfc(data)
        else:
            print("============休闲场摸好牌计算============")
            action, hu_cards = calc_best_cards_by_xxc(data)

        data['cards'] = action
        data["hu_cards"] = hu_cards  # 添加能胡卡牌

        await self.info_log(f"计算有效牌, tid: {data.get('tid')}, uid: {data.get('uid')}, action: {action}")
        print('当前预测摸牌ID为: {}'.format(data['self']))
        print('AI输出预测摸牌: {}'.format(action))
        print()
        print("%##############<< 预测下一位摸好牌 >>##################%")
        await self.send_child_game(data, action)

    async def start_server(self):
        await self.handle_task()

    @staticmethod
    def share_server(server_name):
        """
        发送服务
        """
        return MahjongHandler(server_name)

    @staticmethod
    def calc_remain_cards(curr_hand_cards, remain_cards):
        """
        统计剩余卡牌
        """
        # 出牌、碰牌、杠牌已经减去，此处不再计算
        if not remain_cards:
            return None
        new_remain_cards = {int(key): value for key, value in remain_cards.items()}
        cards_dict = Counter(curr_hand_cards)
        for card, nums in cards_dict.items():
            if new_remain_cards.get(card, 0):
                new_remain_cards[card] -= nums

        print("%######%% 计算剩余卡牌: {}".format(new_remain_cards))

        return new_remain_cards

    def calc_xxc_actions(self, data):
        """
        计算休闲场出牌
        """
        print("============计算休闲场出牌============")
        mg = MoveGenerator()
        # 计算剩余的卡牌并更新卡牌属性
        remain_cards = self.calc_remain_cards(data.get("curr_hand_cards") or [], data.get("remain_cards") or {})
        mg.update_attr(
            data.get("curr_hand_cards"),
            data.get("piles"),
            data.get("left_count"),
            data.get("others_hand_cards"),
            remain_cards
        )
        action = mg.calc_xts_by_max_hu_type(data["ting_list"], data["others_cards_and_piles"])

        return action

    def calc_hfc_actions(self, data):
        """
        计算话费场出牌
        """
        print("============计算话费场出牌============")
        mg = HJMoveGenerator()
        # 计算剩余的卡牌并更新卡牌属性
        remain_cards = self.calc_remain_cards(data.get("curr_hand_cards") or [], data.get("remain_cards") or {})
        mg.update_attr(
            data.get("curr_hand_cards"),
            data.get("piles"),
            data.get("left_count"),
            data.get("others_hand_cards"),
            remain_cards
        )

        action = mg.calc_xts_by_max_hu_type(data["ting_list"], data["others_cards_and_piles"])
        return action

    @staticmethod
    def match_robot(mg, data):
        """
        不同场次对应不同等级机器人
        """
        # match_id=1，麻将雀神赛预赛
        match_id = data.get("match_id", 0)
        print("#@@@@= 麻将匹配场次ID =@@@@#: {}".format(match_id))
        if not match_id or match_id in (1, 5):
            return mg.calc_xts_by_max_hu_type(data["ting_list"], data["others_cards_and_piles"])

        # match_id=2，一元话费极速赛
        # match_id=3，五元话费争夺赛
        # match_id=4，十元话费争夺赛
        if match_id < 5:
            if match_id == 4:
                return mg.calc_xts_by_normal_hu_type()
            return mg.calc_xts_by_normal_hu_type_old()

        # match_id=5，五十元话费争夺赛
        return mg.calc_xts_by_max_hu_type_no_dian_pao(data["ting_list"], data["others_cards_and_piles"])