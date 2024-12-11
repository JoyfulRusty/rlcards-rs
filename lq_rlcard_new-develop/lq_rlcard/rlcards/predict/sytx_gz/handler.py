# -*- coding: utf-8 -*-

import time

from base.base_server import BaseServer
from rlcards.const.sytx_gz.const import ActionType
from rlcards.games.sytx_gz.allocat import get_better_combine_by_3
from rlcards.predict.sytx_gz.run import predict_farmer, predict_landlord


class SyHandler(BaseServer):
	"""
	水鱼预测处理函数
	"""
	def __init__(self, server_name):
		super().__init__(server_name)

	async def cal_farmer_sp(self, data: dict):
		"""
		todo: 计算闲家撒扑

		预测结果为：
			1.闲密庄开
			2.闲密庄认
		"""
		await self.info_log(data)  # 记录日志

		print("计算时间: {}, 从队列中读取数据: {}".format(self.record_calc_time(), data))
		print("######-> AI开始计算闲家，庄家选择预测动作 <-######")

		# 闲家撒扑，只能选择三种操作[撒扑、密、强攻]
		action, action_command = predict_farmer(data)

		send_data = self.parse_receive_data(data)

		send_data["combine_cards"] = get_better_combine_by_3(data["bright_cards"])
		send_data['sa_pu_op'] = self.cal_ai_sp_command(action)
		# send_data["sa_pu_sort"] = ...  sa_pu_sort未使用
		send_data['ai_cmd'] = action_command

		print("输出当前预测后的数据: ", send_data)
		print("输出预测动作: ", action, "动作调用命令: ", action_command)
		print()
		print("%##############<< 预测下一位玩家出牌动作 >>##################%")

		await self.send_child_game(send_data)

	async def cal_farmer_mi(self, data: dict):
		"""
		todo: 闲家密，庄家选择预测动作

		预测结果为：
			1.闲密庄开
			2.闲密庄认
		"""
		await self.info_log(data)  # 记录日志

		print("计算时间: {}, 从队列中读取数据: {}".format(self.record_calc_time(), data))
		print("######-> AI开始计算闲家密，庄家选择预测动作 <-######")

		# todo: 闲家密，庄家只能开认
		action, action_command = predict_landlord(data)

		send_data = self.parse_receive_data(data)

		send_data['sa_pu_op'] = self.cal_ai_mi_command(action)
		send_data['ai_cmd'] = action_command

		print("输出当前预测后的数据: ", send_data)
		print("输出预测动作: ", action, "动作调用命令: ", action_command)
		print()
		print("%##############<< 预测下一位玩家出牌动作 >>##################%")

		await self.send_child_game(send_data)

	async def cal_farmer_lp(self, data: dict):
		"""
		todo: 闲家亮牌，庄家选择预测动作

		预测结果为:
			1.闲亮庄杀
			2.闲亮庄开
		"""
		await self.info_log(data)  # 记录日志

		print("计算时间: {}, 从队列中读取数据: {}".format(self.record_calc_time(), data))
		print("######-> AI开始计算闲家亮牌，庄家选择预测动作 <-######")

		# todo: 闲家亮牌，庄家只能杀走
		action, action_command = predict_landlord(data)

		send_data = self.parse_receive_data(data)

		send_data['sa_pu_op'] = self.cal_ai_lp_command(action)
		send_data['ai_cmd'] = action_command

		print("输出当前预测后的数据: ", send_data)
		print("输出预测动作: ", action, "动作调用命令: ", action_command)
		print()
		print("%##############<< 预测下一位玩家出牌动作 >>##################%")

		await self.send_child_game(send_data)

	async def cal_landlord_zou(self, data: dict):
		"""
		todo: 庄家走，闲家选择预测动作

		预测结果为:
			1.庄走闲信
			2.庄走闲反
		"""
		await self.info_log(data)  # 记录日志

		print("计算时间: {}, 从队列中读取数据: {}".format(self.record_calc_time(), data))
		print("######-> AI开始计算庄家走，闲家选择预测动作 <-######")

		# todo: 庄家走，闲家只能信反
		action, action_command = predict_farmer(data)

		send_data = self.parse_receive_data(data)

		send_data['sa_pu_op'] = self.cal_ai_command(action, data)
		send_data['ai_cmd'] = action_command

		print("输出当前预测后的数据: ", send_data)
		print("输出预测动作: ", action, "动作调用命令: ", action_command)
		print()
		print("%##############<< 预测下一位玩家出牌动作 >>##################%")

		await self.send_child_game(send_data)

	async def cal_landlord_sha(self, data: dict):
		"""
		todo: 庄杀，闲家选择预测动作

		预测结果为:
			1.庄杀闲反
			2.庄杀闲信
		"""
		await self.info_log(data)  # 记录日志

		print("计算时间: {}, 从队列中读取数据: {}".format(self.record_calc_time(), data))
		print("######-> AI开始计算庄家杀，闲家选择预测动作 <-######")

		# todo: 庄家杀，闲家只能杀反
		action, action_command = predict_farmer(data)

		send_data = self.parse_receive_data(data)

		send_data['sa_pu_op'] = self.cal_ai_command(action, data)
		send_data['ai_cmd'] = action_command

		print("输出当前预测后的数据: ", send_data)
		print("输出预测动作: ", action, "动作调用命令: ", action_command)
		print()
		print("%##############<< 预测下一位玩家出牌动作 >>##################%")

		await self.send_child_game(send_data)

	async def start_server(self):
		await self.handle_task()

	@staticmethod
	def share_server(server_name: str):
		"""
		发送服务
		"""
		return SyHandler(server_name)

	@staticmethod
	def record_calc_time():
		"""
		记录当前计算时间
		"""
		return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

	@staticmethod
	def parse_receive_data(data):
		"""
		解析接收数据
		"""
		send_data = {
			"tid": data.pop("tid", 0),
			"uid": data.pop("uid", 0),
			"cmd": data.pop("cmd", 0),
			"res": data.pop("res", 0),
			"target_id": data.pop("target_id", 0),
			"dealer_id": data.pop("dealer_id", 0),
			"server_type": data.pop("server_type", 32),
			"server_index": data.pop("server_index", 1)
		}
		return send_data

	@staticmethod
	def cal_ai_command(action, data):
		"""
		计算AI走杀命令
		"""
		# 杀[8]走[9]
		if data["last_action"] == 8:
			return {ActionType.FAN: 8, ActionType.XIN: 7}[action]
		return {ActionType.FAN: 10, ActionType.XIN: 9}[action]

	@staticmethod
	def cal_ai_sp_command(action):
		"""
		计算AI撒扑命令
		"""
		if action == ActionType.SP:
			return 3  # 亮牌
		return 1 if action == ActionType.QG else 2

	@staticmethod
	def cal_ai_mi_command(action):
		"""
		计算AI密命令
		"""
		return 3 if action == ActionType.XIN else 4

	@staticmethod
	def cal_ai_lp_command(action):
		"""
		计算AI亮牌命令
		"""
		return 1 if action == ActionType.SHA else 2