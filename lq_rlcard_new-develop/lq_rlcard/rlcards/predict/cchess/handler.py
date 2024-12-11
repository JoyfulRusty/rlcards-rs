# -*- coding: utf-8 -*-

import time

from base.base_server import BaseServer
from rlcards.predict.cchess.run import agent


class CCHandler(BaseServer):
	"""
	象棋动作预测函数
	"""
	def __init__(self, server_name):
		super().__init__(server_name)

	async def cal_action(self, data):
		"""
		象棋计算动作
		"""
		await self.info_log(data)  # 日志记录

		print("从队列中读取数据: {}".format(data))
		print("######-> AI开始计算预测动作 <-######")

		# to_fen -> 传入当前棋盘情况
		to_fen = data.get('to_fen', None)
		start_position, end_position = agent.step(to_fen)

		data['start_position'] = start_position
		data['end_position'] = end_position

		print("输出当前预测后的数据: ", data)
		print("输出预测动作: ", action, "动作调用命令: ", action_command)
		print()

		await self.send_child_game(data)

	async def start_server(self):
		await self.handle_task()

	@staticmethod
	def share_server(server_name: str):
		"""
		发送服务
		"""
		return CCHandler(server_name)