# -*- coding: utf-8 -*-

from base.base_server import BaseServer

class SyHandler(BaseServer):
	"""
	水鱼预测处理函数
	"""
	def __init__(self, server_name):
		super().__init__(server_name)

	async def cal_action(self, data: dict):
		"""
		计算动作选择
		"""
		await self.info_log(data)  # 记录日志

		print("从队列中读取数据: {}".format(data))
		print("######-> AI开始计算预测动作 <-######")

		action = ...

		data['cards'] = action

		await self.send_child_game(data, action)

	async def start_server(self):
		await self.handle_task()

	@staticmethod
	def share_server(server_name: str):
		"""
		发送服务
		"""
		return SyHandler(server_name)