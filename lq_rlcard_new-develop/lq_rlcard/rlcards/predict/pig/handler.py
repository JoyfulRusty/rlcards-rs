# -*- coding: utf-8 -*-

from base.base_server import BaseServer
from rlcards.predict.pig.predict import predict_action


class PigHandler(BaseServer):
    """
    拱猪处理函数
    """

    def __init__(self, server_name):
        super().__init__(server_name)

    async def cal_action(self, data):
        """
        计算出牌动作
        """
        await self.info_log(data)  # 日志记录

        print("从队列中读取数据: ", data)
        print("######-> AI开始计算预测动作 <-######")

        action = predict_action(data)  # 开始计算动作

        data['card'] = action

        print('预测玩家ID为: ', data["player_position"])
        print('AI输出预测打牌动作: ', action)
        print()
        print("%##############<< 预测下一位玩家出牌动作 >>##################%")

        await self.send_child_game(data, action)

    async def start_server(self):
        await self.handle_task()

    @staticmethod
    def share_server(server_name):
        return PigHandler(server_name)