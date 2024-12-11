# -*- coding: utf-8 -*-

import asyncio

from typing import Optional
from public.const import LOG_DEBUG
from public.logs import Logger, log_handle


class BaseLogOut():
    """
    基础日志输出类
    """
    def __init__(self, server_name):
        """
        初始化参数
        """
        self.server_name = server_name
        self.log_debug = LOG_DEBUG
        self.__logger: Optional[Logger] = None

    async def logger(self):
        """
        日志方法
        """
        if not isinstance(self.__logger, Logger):
            self.__logger = await log_handle.get_logger(self.server_name)
        return self.__logger

    async def info_log(self, *msg):
        """
        日志信息
        """
        if self.log_debug:
            print(msg)
        else:
            if not self.__logger:
                await asyncio.create_task(self.logger())  # 此时loop已经在运行了，往该loop中添加协程任务即可
            await self.__logger.info(msg)

    async def error_log(self, data):
        """
        错误日志
        """
        name = f"{self.server_name}_error"
        await log_handle.fail_log(data, folder_name=name)