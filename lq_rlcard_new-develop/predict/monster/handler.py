# -*- coding: utf-8 -*-

import time

import torch

from base.base_server import BaseServer
from utils.utiltools import get_datetime_fmt
from predict.monster.models import dyg_model_dict
from public.proto.py_pb2.c2s_msg import play_card_model
from reinforce.games.monster_v2.state import MonsterState
from base.base_const import RobotCmdMethods, ServiceEnum, CmdRoomMethods

class MonsterHandler(BaseServer):
    """
    打妖怪-机器人
    """

    def __init__(self, server_name):
        super().__init__(server_name)
        self.model_dict = dyg_model_dict
        self.state = MonsterState()
        self.add_handlers({
            RobotCmdMethods.CAL_ACTION.val: self.cal_action,
        })

    async def cal_action(self, uid, req_data: dict):
        """
        预测动作
        """
        start_time = time.time()

        await self.info_log(f"######-> 模型请求ID-[req_model_id]: {req_data.get('req_model_id', 0)} <-######")
        await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列[rabbitmq-queue]中读取数据: {req_data}")
        await self.info_log(f"######-> AI开始计算预测动作 <-######")

        action = self.calc_model_action(req_data)
        await self.info_log(f'预测玩家UID为: {uid}')
        await self.info_log(f'输出预测动作: {action}')
        # 序列化预测出牌数据为proto buf -> b'\x08i'
        play_card_model.cards[:] = []  # 清空数据
        play_card_model.cards.append(action)
        cards = play_card_model.SerializeToString()
        if action == 621:
            await self.pick_cards(uid, cards)
        else:
            await self.play_cards(uid, cards)
        end_time = time.time()
        await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
        await self.info_log()
        await self.info_log(f"%##############<< 预测下一位玩家出牌动作 >>##################%")
        # await self.cs2cs_by_rmq(ServiceEnum.C_MONSTER, RobotCmdMethods.CAL_ACTION, uid=uid, msg=predict_infos)

    async def play_cards(self, uid, data):
        """
        机器人出牌
        """
        await self.cs2cs_by_rmq(ServiceEnum.C_MONSTER, CmdRoomMethods.PLAY_CARDS, uid=uid, msg=data)

    async def pick_cards(self, uid, data):
        """
        机器人捡牌
        """
        await self.cs2cs_by_rmq(ServiceEnum.C_MONSTER, CmdRoomMethods.PICK_CARDS, uid=uid, msg=data)

    def calc_model_action(self, req_data: dict):
        """
        计算模型预测动作
        """
        start_time = time.time()
        self.state.init_attrs(
            seat_id=req_data.get('seat_id', 0),
            hand_cards=req_data.get('hand_cards', []),
            played_cards=req_data.get('played_cards', []),
            last_action=req_data.get('last_action', 0),
            legal_actions=req_data.get('legal_actions', []),
            round_cards=req_data.get('round_cards', []),
            remain_cards=req_data.get('remain_cards', []),
            action_history=req_data.get('action_history', []),
            other_left_cards=req_data.get('other_left_cards', []),
            other_played_cards=req_data.get('other_played_cards', []),
            bust_infos=req_data.get('bust_infos', []),
        )
        position = req_data.get("position", "down")
        model = self.model_dict.get(position)
        z_batch, x_batch, legal_actions = self.format_obs(self.state.get_obs())
        if len(legal_actions) == 1:
            return legal_actions[-1]
        action_idx = model.predict(z_batch, x_batch)
        end_time = time.time()
        print("===================== 输出模型预测结果 =====================")
        print("seat_id: ", req_data.get('seat_id', 0))
        print("Legal actions: ", legal_actions)
        print("Prediction action: ", legal_actions[action_idx])
        print("Consuming time: ", end_time - start_time)
        print()
        return legal_actions[action_idx]

    @staticmethod
    def format_obs(obs):
        """
        将观测数据转换为神经网络输入格式
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_batch = torch.from_numpy(obs['x_batch']).to(device)
        z_batch = torch.from_numpy(obs['z_batch']).to(device)
        return z_batch, x_batch, obs['legal_actions']