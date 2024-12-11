import asyncio
import logging
from aiologger import Logger
from aiologger.formatters.base import Formatter
from aiologger.handlers.files import AsyncTimedRotatingFileHandler, RolloverInterval
from pathlib import Path
from public.const import OUTPUT_PATH

LOG_FORMAT = "%(asctime)s - %(process)s:%(lineno)d - %(levelname)s - %(message)s"


class LogHandler():
    __FILES_HANDLE = {}

    def __init__(self):
        self.format = LOG_FORMAT

    async def get_logger(self, name: str = "log", when=RolloverInterval.HOURS, level=logging.DEBUG) -> Logger:
        """异步日志，按日期切割"""
        logger = self.__FILES_HANDLE.get(name)
        if not logger:
            logger: Logger = Logger(name=name, level=level)
            logfile_path = Path(OUTPUT_PATH, name, f"{name}.log")
            logfile_path.parent.mkdir(parents=True, exist_ok=True)
            logfile_path.touch(exist_ok=True)
            atr_file_handler = AsyncTimedRotatingFileHandler(
                filename=str(logfile_path),
                when=when,
                backup_count=24,
                encoding="utf8"
            )
            formatter = Formatter(self.format)
            atr_file_handler.formatter = formatter
            logger.add_handler(atr_file_handler)
            self.__FILES_HANDLE[name] = logger
        return logger

    async def fail_log(self, msg: str, **kwargs):
        """启动日志"""
        name = kwargs.get("folder_name") or "failed_info"
        log_fail: Logger = await self.get_logger(name)
        await log_fail.info(msg)

    async def test(self, name: str, msg: str):
        """启动日志"""
        log_sys: Logger = self.__FILES_HANDLE.get(name)
        await log_sys.info(msg)

    async def log_shutdown(self, name):
        """停掉日志"""
        log_sys: Logger = await self.get_logger(name)
        await log_sys.shutdown()


log_handle = LogHandler()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    log_name = "system"
    loop.run_until_complete(log_handle.test(log_name, "服务启动成功"))
    loop.run_until_complete(log_handle.log_shutdown(log_name))
