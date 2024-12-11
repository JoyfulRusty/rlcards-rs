# -*- coding: utf-8 -*-

import time

from base.base_server import BaseServer
from rlcards.predict.monster.run import predict_action


class MonsterHandler(BaseServer):
    """
    打妖怪处理函数
    """

    def __init__(self, server_name):
        super().__init__(server_name)

    async def cal_action(self, data):
        """
        预测动作
        """
        start_time = time.time()
        await self.info_log(data)  # 日志记录

        print("从队列中读取数据: ", data)

        # 预测动作
        print("######-> AI开始计算预测动作 <-######")
        action = predict_action(data)
        data["action"] = 99 if action == 'PICK_CARDS' else action
        print('预测玩家ID为: ', data['self'])
        print('AI输出预测打牌动作: ', action)

        end_time = time.time()
        print(f"%##############输出计算耗时: {end_time - start_time}")
        print()

        print("%##############<< 预测下一位玩家出牌动作 >>##################%")


        await self.send_child_game(data, action)

    async def start_server(self):
        await self.handle_task()

    @staticmethod
    def share_server(server_name):
        return MonsterHandler(server_name)