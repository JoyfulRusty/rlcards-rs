import asyncio

from utils import utils
from typing import Optional
from public.logs import Logger
from public.const import ChannelType
from base import BaseLogOut, BaseRedis


class BaseServer(BaseRedis, BaseLogOut):
    def __init__(self, server_name):
        BaseRedis.__init__(self)
        BaseLogOut.__init__(self, server_name)
        self.__logger: Optional[Logger] = None
        self.log_debug = False

    async def __read_task(self):
        """
        循环独取队列
        """
        while 1:
            try:
                data = await self.read_queue(self.server_name)
                if data:
                    data = await utils.to_dict(data)
                    method_str = data.get("method") or None
                    if not hasattr(self, method_str):
                        continue
                    method = getattr(self, method_str)
                    await method(data)
            except Exception as e:
                await self.error_log(f"打印异常信息: {e}")
                await self.close()
                break
            finally:
                await asyncio.sleep(0.01)

    async def handle_task(self):
        """ 处理任务 """
        await self.__read_task()

    @staticmethod
    def share_server(server_name):
        ...

    def start_server(self):
        raise TypeError(f"子类请重写{self.start_server.__name__}方法")

    async def send_child_game(self, data: dict, code=0):
        """ 发送消息到子游戏 """
        sid = 1
        server_type = data.pop("server_type")
        cmd = data.pop("cmd")
        server_index = data.pop("server_index", 1)
        uid = data.get("uid")
        channel = ChannelType.CHANNEL_GATEWAY_CHILD
        if server_type == 32:
            channel = "channel_gateway_child_ddz"
        message = self.format_channel_struct(sid, server_type, cmd, uid, data, code, server_index)
        await self.publish(channel, utils.json_dumps(message))

    @staticmethod
    def format_channel_struct(sid: int, service: int, cmd: int, uid: int, data, code, server_index=1) -> list:
        """ 格式化要发送到频道的消息 """
        return [sid, service, cmd, uid, data, code, server_index]

    @staticmethod
    def read_channel_struct(message: list):
        """
        读取通道的数据结构
        :param message: 消息
        :return: 返回读取内容
        """
        if 6 == len(message):
            sid, service, cmd, uid, msg, code = message
            return sid, service, cmd, uid, msg, code, 1
        if 7 == len(message):
            sid, service, cmd, uid, msg, code, service_index = message
            return sid, service, cmd, uid, msg, code, service_index