# -*- coding: utf-8 -*-

import time

from predict.ddz.state import Model
from utils.utiltools import get_datetime_fmt

from base.base_server import BaseServer
from base.base_const import RobotCmdMethods, ServiceEnum, CmdRoomMethods

from public.proto.py_pb2.c2s_msg import do_bid_model, play_card_model, do_redouble_model


class LandlordsHandler(BaseServer):
	"""
	斗地主休闲场-机器人
	"""

	def __init__(self, server_name):
		"""
		初始化参数
		"""
		super().__init__(server_name)
		self.model = Model()
		self.add_handlers({
			RobotCmdMethods.CAL_ACTION.val: self.cal_action,
			RobotCmdMethods.CAL_BID.val: self.cal_bid,
			RobotCmdMethods.CAL_CHAN.val: self.cal_chan,
			RobotCmdMethods.CAL_DO_CHAN.val: self.cal_do_chan,
			RobotCmdMethods.CAL_EXCHANGE_THREE.val: self.cal_exchange_three,
			RobotCmdMethods.CAL_EXCHANGE_FIVE.val: self.cal_exchange_five,
		})

	async def cal_action(self, uid, req_data: dict):
		"""
		计算出牌
		"""
		start_time = time.time()
		await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列[rabbitmq-queue]中读取数据: {req_data}")
		await self.info_log("######-> AI开始计算预测 [出牌] 动作 <-######")

		actions = self.model.cal_play_card(uid=uid, req_data=req_data)
		play_card_model.cards[:] = []  # 清空数据
		play_card_model.cards.extend(actions)
		cards = play_card_model.SerializeToString()
		await self.info_log(f'预测玩家UID为: {uid}')
		await self.info_log(f'输出预测动作: {actions}')

		end_time = time.time()
		await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
		await self.info_log()
		await self.info_log("%##############<< 预测下一位玩家动作 >>##################%")
		await self.cs2cs_by_rmq(ServiceEnum.C_LANDLORDS, CmdRoomMethods.PLAY_CARDS, uid=uid, msg=cards)

	async def cal_bid(self, uid, req_data: dict):
		"""
		计算叫分
		"""
		start_time = time.time()
		await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列[rabbitmq-queue]中读取数据: {req_data}")
		await self.info_log("######-> AI开始计算预测 [叫分] 动作 <-######")

		is_bxp = req_data.get("is_bxp", False)
		can_bids = req_data.get("can_bids", [])
		curr_hand_cards = req_data.get("cards", [])
		scores = self.model.cal_play_bid(hand_cards=curr_hand_cards, can_bids=can_bids, is_bxp=is_bxp)
		do_bid_model.bid_score = scores
		do_bid_model.seat_id = uid
		bid_score = do_bid_model.SerializeToString()
		await self.info_log(f'预测玩家UID为: {uid}')
		await self.info_log(f'输出预测叫分: {scores}')

		end_time = time.time()
		await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
		await self.info_log()
		await self.info_log("%##############<< 预测下一位玩家动作 >>##################%")
		await self.cs2cs_by_rmq(ServiceEnum.C_LANDLORDS, CmdRoomMethods.DO_BID, uid=uid, msg=bid_score)

	async def cal_chan(self, uid, req_data: dict):
		"""
		铲
		"""
		start_time = time.time()
		await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列中读取数据: {req_data}")
		await self.info_log("######-> AI开始计算预测 [铲] 动作 <-######")

		curr_hand_cards = req_data.get("cards", [])
		scores = self.model.cal_play_chan(hand_cards=curr_hand_cards)
		do_redouble_model.score = 0 if not scores else 1
		redouble_score = do_redouble_model.SerializeToString()
		await self.info_log(f'预测玩家UID为: {uid}')
		await self.info_log(f'输出预测动作: {scores}')

		end_time = time.time()
		await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
		await self.info_log()
		await self.info_log("%##############<< 预测下一位玩家动作 >>##################%")
		await self.cs2cs_by_rmq(ServiceEnum.C_LANDLORDS, CmdRoomMethods.DO_REDOUBLE, uid=uid, msg=redouble_score)

	async def cal_do_chan(self, uid, req_data: dict):
		"""
		反铲
		"""
		start_time = time.time()
		await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列中读取数据: {req_data}")
		await self.info_log("######-> AI开始计算预测 [反铲] 动作 <-######")

		curr_hand_cards = req_data.get("cards", [])
		scores = self.model.cal_play_chan(hand_cards=curr_hand_cards, chan_threshold=0.6)
		do_redouble_model.score = 0 if not scores else 1
		do_redouble_score = do_redouble_model.SerializeToString()
		await self.info_log(f'预测玩家UID为: {uid}')
		await self.info_log(f'输出预测动作: {scores}')

		end_time = time.time()
		await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
		await self.info_log()
		await self.info_log("%##############<< 预测下一位玩家动作 >>##################%")
		await self.cs2cs_by_rmq(ServiceEnum.C_LANDLORDS, CmdRoomMethods.DO_RE_REDOUBLE, uid=uid, msg=do_redouble_score)

	async def cal_exchange_three(self, uid, req_data: dict):
		"""
		计算换三张
		"""
		start_time = time.time()
		await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列中读取数据: {req_data}")
		await self.info_log("######-> AI开始计算预测 [换三张] 动作 <-######")

		curr_hand_cards = req_data.get("cards", [])
		actions = await self.model.exchange_in_three(cards=curr_hand_cards)
		await self.info_log(f'预测玩家UID为: {uid}')
		await self.info_log(f'输出预测动作: {actions}')

		end_time = time.time()
		await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
		await self.info_log()
		await self.info_log("%##############<< 预测下一位玩家动作 >>##################%")
		await self.cs2cs_by_rmq(ServiceEnum.C_LANDLORDS, CmdRoomMethods.PLAY_CARDS, uid=uid, msg=actions)

	async def cal_exchange_five(self, uid, req_data: dict):
		"""
		计算换五张
		"""
		start_time = time.time()
		await self.info_log(f"计算时间: {get_datetime_fmt()}, 从队列中读取数据: {req_data}")
		await self.info_log("######-> AI开始计算预测 [换五张] 动作 <-######")

		curr_hand_cards = req_data.get("cards", [])
		actions = self.model.exchange_in_five(cards=curr_hand_cards)
		await self.info_log(f'预测玩家UID为: {uid}')
		await self.info_log(f'输出预测动作: {actions}')

		end_time = time.time()
		await self.info_log(f"%###### 输出计算耗时 ######%: {end_time - start_time}")
		await self.info_log()
		await self.info_log("%##############<< 预测下一位玩家动作 >>##################%")
		await self.cs2cs_by_rmq(ServiceEnum.C_LANDLORDS, CmdRoomMethods.PLAY_CARDS, uid=uid, msg=actions)