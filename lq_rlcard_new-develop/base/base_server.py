# -*- coding: utf-8 -*-

import asyncio

from typing import AnyStr
from aio_pika.abc import AbstractIncomingMessage

from base.base_const import ServiceEnum
from base.base_service import BaseService
from utils.utiltools import parse_inner_msg, to_dict


class BaseServer(BaseService):
	"""
	基础类服务
	"""
	__slots__ = ("__listener_obj", )

	def __init__(self, server_name):
		"""
		初始化
		:param server_name: 服务名称
		"""
		super().__init__(server_name)
		self.__listener_obj = None

	async def __read_task(self):
		"""
		rabbitmq循环读取
		"""
		await self.info_log(f"init read task")
		try:
			async for message in self.__listener_obj.listen():
				try:
					await self.info_log("listen message ->: ", message)
					data = message["data"]  # 订阅成功返回1（忽略）
					await self.info_log(f"收到消息：{message}")
					if data == 1:
						continue
					asyncio.create_task(self.receive_data_callback(data))
				except Exception as e:
					await self.error_log(f"message callback error:{e}")
		except asyncio.exceptions.CancelledError:
			await self.on_signal_stop("xxx")
		except Exception as data:
			await self.error_log(f"read task error: {data}")
			await asyncio.sleep(0.5)

	async def receive_data_callback(self, data: AnyStr):
		"""
		收到消息回调
		:param data: 数据
		"""
		(cmd, uid), data = parse_inner_msg(data)
		# 不存在cmd或data为空则直接return
		if not cmd or not data:
			await self.error_log(f"receive_data_callback error: cmd: {cmd}, uid: {uid}, data: {data}")
			return
		try:
			data = await to_dict(data=data)
		except Exception as err:
			await self.error_log(f"parse receive_data_callback error: {err}, data: {data}")
			return
		await self.call_handler(cmd, uid, data)

	async def call_handler(self, cmd, uid, data: AnyStr):
		"""
		调用处理函数
		:param cmd: 命令
		:param uid: 用户id
		:param data: 数据
		"""
		await self.service(cmd, uid, data)

	async def __consume_rmq(self, exchange_name=''):
		"""
		消费rabbitmq
		:param exchange_name: 交换名称
		"""
		exchange_name = exchange_name or self.listen_channel[0]
		que_name = self.queue_name
		await self.conf.rmq.consume(exchange_name, que_name, que_name, on_message=self.on_message)

	async def on_message(self, message: AbstractIncomingMessage) -> None:
		"""
		rmq消息回调
		:param message: AbstractIncomingMessage消息
		"""
		try:
			await asyncio.create_task(self.receive_data_callback(message.body))
			message and await message.ack()  # todo: 后续根据需要确认消息
		except Exception as e:
			await self.error_log(f"on_message error:{e}")

	async def on_signal_stop(self, *args):
		"""
		服务关闭时触发
		:param args: 参数
		"""
		await self.info_log(f"{self.service_name}服务关闭")
		self.conf.rmq.close()
		await self.__listener_obj.close()

	async def __init_listen_channel(self):
		"""
		初始化监听通道
		"""
		assert self.listen_channel
		if self.__listener_obj:
			await self.__listener_obj.close()
			self.__listener_obj = None
		await asyncio.sleep(0)
		while 1:
			try:
				obj = await self.conf.rds.pub_sub(self.listen_channel)  # 订阅与发布系统状态
				self.__listener_obj = obj
				break
			except Exception as data:
				await self.error_log(f"sid {self.server_id} init_listen_channel error: {str(data)}")
				await asyncio.sleep(2)

	async def start_server(self):
		"""
		启动子游戏服务
		"""
		await self.conf.rds.init_rds_pool()  # 初始化redis
		await self.conf.rmq.init_rmq_pool()  # 初始化rmq
		await self.conf.msql.init_msql_pool()  # 初始化数据库连接池
		await self.__init_listen_channel()  # 初始化监听频道
		await self.__consume_rmq()  # rmq消费

	async def cs2cs_by_rmq(self, cs_type: ServiceEnum, c_code, uid=1, msg=None):
		"""
		通过rmq推送机器人操作到子游戏
		:param cs_type: 服务类型
		:param c_code: 子游戏code
		:param uid: 用户id
		:param msg: 消息
		"""
		await self.conf.rmq.cs2cs_rmp(cs_type, c_code, uid, msg)

	@classmethod
	def share_server(cls, server_name):
		"""
		共享服务
		:param server_name: 服务名
		"""
		return cls(server_name)