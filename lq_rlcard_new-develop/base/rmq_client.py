# -*- coding: utf-8 -*-

import asyncio

from typing import AnyStr
from nsanic.libs.tool import json_encode

from aio_pika.pool import Pool
from aio_pika.abc import AbstractRobustConnection, ExchangeType
from aio_pika import DeliveryMode, Message, connect_robust, Channel

from base.base_const import RmqGameChannel, ServiceEnum
from utils.utiltools import pack_inner_msg


class RmqClient:
    """
    rabbitmq client
    """
    CONN_MAP = {}

    def __init__(self, conf, logs=None):
        """
        :param conf: 连接配置
        :param logs: 日志
        """
        self.__ping = 30
        self.__conf = conf
        self.__logs = logs
        self.__conn_pool = None  # 连接池：从连接池中获取连接频道
        self.__channel_conn_pool = None  # 频道连接池
        self.__queue = None
        self.__exchange = None
        self.__fut = None

    @classmethod
    def init(cls, conf: dict, logs=None):
        """
        :param conf: 连接配置
        :param logs: 日志
        """
        if not all(key in conf for key in ['host', 'port', 'login']):
            raise Exception('conf must contain host, port, login')
        conf_name = f"{conf['host']}-{conf['port']}-{conf['login']}"
        clt = cls.CONN_MAP.get(conf_name)
        if not clt:
            clt = cls(conf, logs=logs)
            cls.CONN_MAP[conf_name] = clt
        return clt

    async def init_rmq_pool(self):
        """
        初始化连接池
        """
        try:
            if not self.__conn_pool:
                self.__conn_pool: Pool = Pool(self.get_connection, max_size=2)
                await self.__logs.info_log(f"rabbitmq init connection pool success: {self.__conn_pool}")
            if not self.__channel_conn_pool:
                self.__channel_conn_pool = Pool(self.get_channel, max_size=10)
                await self.__logs.info_log(f"rabbitmq init channel pool success: {self.__channel_conn_pool}")
        except Exception as err:
            await self.__logs.error_log(f"init rabbitmq connection and channel pool error: {err}")

    async def get_connection(self) -> AbstractRobustConnection:
        """
        获取连接
        """
        try:
            return await connect_robust(**self.__conf)
        except Exception as err:
            await self.__logs.error_log(f"rabbitmq get_connection: {err}")

    async def get_channel(self) -> Channel:
        """
        获取频道
        """
        try:
            async with self.__conn_pool.acquire() as connection:
                return await connection.channel()
        except Exception as err:
            await self.__logs.error_log(f"rabbitmq get_channel: {err}")

    async def del_queue(self, que_name: str):
        """
        删除队列
        :param que_name: 队列名称
        """
        try:
            async with self.__channel_conn_pool.acquire() as channel:
                queue = await channel.get_queue(que_name)
                await queue.delete()
        except Exception as err:
            await self.__logs.error_log(f"rabbitmq del_queue: {err}")

    @staticmethod
    async def del_exchange(exchange):
        """
        删除交换器
        :param exchange: 交换器对象
        """
        try:
            hasattr(exchange, 'delete') and await exchange.delete()
        except Exception as err:
            print(f"rabbitmq del_exchange: {err}")

    def close(self):
        """
        关闭连接
        """
        try:
            if self.__fut:
                self.__fut.set_result("close")
        except Exception as err:
            print(f"rabbitmq close: {err}")

    async def consume(self, exchange_name, que_name, routing_key='', extra_key="", on_message=None, que_auto_del=False) -> None:
        """
        消费消息

        :param exchange_name: 交换器名称
        :param que_name: 队列名称
        :param routing_key: 路由键
        :param extra_key: 额外的路由键
        :param on_message: 消息处理函数
        :param que_auto_del: 队列是否自动删除
        """
        async with self.__channel_conn_pool.acquire() as channel:  # type: Channel
            # await channel.set_qos(10)  # 只有当这10条消息中的至少一条被确认后，RabbitMQ才会发送更多的消息。
            match_exchange = await channel.declare_exchange(
                exchange_name, ExchangeType.DIRECT, durable=True
            )
            # 声明队列
            queue = await channel.declare_queue(
                que_name, auto_delete=que_auto_del
            )

            # Binding the queue to the exchange
            # 将队列绑定到交换器
            await queue.bind(match_exchange, routing_key=routing_key)
            extra_key and await queue.bind(match_exchange, routing_key=extra_key)

            # Start listening the queue
            # 监听队列
            await queue.consume(on_message)

            print(" [*] Waiting for msgs. To exit press CTRL+C")
            self.__fut = asyncio.Future()
            await self.__fut

            # 使用迭代器读取消息
            # async with queue.iterator() as queue_iter:
            #     async for message in queue_iter:
            #         print(message)
            #         await message.ack()

    async def publish(self, msg: bytes, exchange_name, routing_key="", exp=None) -> None:
        """
        发布消息

        :param msg: 消息内容
        :param exchange_name: 交换器名称
        :param routing_key: 路由键
        :param exp: 过期时间 seconds or (or datetime or timedelta)
        """
        async with self.__channel_conn_pool.acquire() as channel:  # type: Channel
            match_exchange = await channel.declare_exchange(
                exchange_name, ExchangeType.DIRECT, durable=True
            )
            msg_obj = Message(
                msg,
                expiration=exp,
                delivery_mode=DeliveryMode.PERSISTENT,  # 标记持久消息
            )
            await match_exchange.publish(
                msg_obj,
                routing_key=routing_key,
            )

    async def cs2cs_rmp(
            self, target_cs: ServiceEnum, cmd, uid=1, msg: AnyStr = None, r_key="", exp=30):
        """
        子服务间（进程间）发送消息通过rmq（主要用在匹配）
        r_key: 指定路由键，不指定取c_type.phrase
        exp: 过期时间 seconds or (or datetime or timedelta)
        uid = 1 时默认为不需要uid（打包消息时不能小于等于0）
        msg: 传递json
        """
        try:
            if not isinstance(msg, bytes):
                msg = json_encode(msg, u_byte=True)
            pack_data = pack_inner_msg(cmd, uid, msg)
            await self.push_data_by_rmq(target_cs, pack_data, r_key, exp=exp)
        except Exception as err:
            await self.__logs.error_log(f"cs2cs_rmp error {err}")

    async def push_data_by_rmq(self, target_cs: ServiceEnum, msg: bytes, r_key="", exp=None, log_fun=None):
        """
        通过rmq推送消息
        r_key: 指定路由键，不指定取c_type.phrase
        target_cs: 目标服务
        此处使用直连交换器
        """
        exchange_name = f"{RmqGameChannel.C_SERVICES}_{target_cs.val}"
        routing_key = r_key or f"{target_cs.phrase}_{target_cs.val}"
        try:
            await self.publish(msg, exchange_name, routing_key, exp=exp)
        except Exception as err:
            await self.__logs.error_log(f"push_data_by_rmq error {err}")