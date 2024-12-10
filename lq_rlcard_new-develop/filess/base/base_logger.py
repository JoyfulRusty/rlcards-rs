# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import asyncio
import datetime

from typing import List, Optional

from aiologger import Logger
from aiologger.records import LogRecord
from aiologger.utils import get_running_loop
from aiologger.formatters.base import Formatter

from utils.meta_class import SingleTon

from pathlib import Path
from public.const import OUTPUT_PATH, LOG_DEBUG

from aiologger.handlers.files import (
	ONE_DAY_IN_SECONDS,
	ONE_MINUTE_IN_SECONDS,
	ONE_HOUR_IN_SECONDS,
	RolloverInterval,
	BaseAsyncRotatingFileHandler,
)

LOG_FORMAT = "%(asctime)s - %(process)s:%(lineno)d - %(levelname)s - %(message)s"

# todo: (重构)用于记录到文件的处理程序，以特定的时间间隔轮换日志文件

class AsyncTimedRotatingFileHandler(BaseAsyncRotatingFileHandler):
	"""
	用于记录到文件的处理程序，以特定的时间间隔轮换日志文件
	如果backup_count > 0，则翻转完成后，不会保留超过backup_count的文件，并将最旧的文件将被删除
	| when       | at_time behavior                                       |
	|------------|--------------------------------------------------------|
	| SECONDS    | at_time will be ignored                                |
	| MINUTES    | -- // --                                               |
	| HOURS      | -- // --                                               |
	| DAYS       | at_time will be IGNORED. See also MIDNIGHT             |
	| MONDAYS    | rotation happens every WEEK on MONDAY at ${at_time}    |
	| TUESDAYS   | rotation happens every WEEK on TUESDAY at ${at_time}   |
	| WEDNESDAYS | rotation happens every WEEK on WEDNESDAY at ${at_time} |
	| THURSDAYS  | rotation happens every WEEK on THURSDAY at ${at_time}  |
	| FRIDAYS    | rotation happens every WEEK on FRIDAY at ${at_time}    |
	| SATURDAYS  | rotation happens every WEEK on SATURDAY at ${at_time}  |
	| SUNDAYS    | rotation happens every WEEK on SUNDAY at ${at_time}    |
	| MIDNIGHT   | rotation happens every DAY at ${at_time}               |
	"""

	def __init__(
			self,
			filename: str,
			when: RolloverInterval = RolloverInterval.HOURS,
			interval: int = 1,
			backup_count: int = 0,
			encoding: str = None,
			utc: bool = False,
			at_time: datetime.time = None) -> None:
		"""
		:param filename: 文件名
		:param when: 翻转间隔
		:param interval: 间隔时间
		:param backup_count: 备份文件数量
		:param encoding: 文件编码
		:param utc: 是否使用UTC时间
		:param at_time: 翻转时间
		"""
		super().__init__(filename=filename, mode="a", encoding=encoding)
		self.utc = utc
		self.at_time = at_time
		self.when = when.upper()
		self.backup_count = backup_count

		# 计算实际翻转间隔，即翻转之间的秒数，还可用设置发生翻转时使用的文件名后缀，当前支持的时间事件:
		# S - Seconds
		# M - Minutes
		# H - Hours
		# D - Days
		# {0 - 6}
		# 在某一天滚动； 0 - 星期一 when说明符的大小写并不重要；小写或大写都可以
		if self.when == RolloverInterval.SECONDS:
			self.interval = 1  # one second
			self.suffix = "%Y-%m-%d_%H-%M-%S"
			ext_match = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(\.\w+)?$"
		elif self.when == RolloverInterval.MINUTES:
			self.interval = ONE_MINUTE_IN_SECONDS  # one minute
			self.suffix = "%Y-%m-%d_%H-%M"
			ext_match = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(\.\w+)?$"
		elif self.when == RolloverInterval.HOURS:
			self.interval = ONE_HOUR_IN_SECONDS  # one hour
			self.suffix = "%Y-%m-%d_%H"
			ext_match = r"^\d{4}-\d{2}-\d{2}_\d{2}(\.\w+)?$"
		elif self.when == RolloverInterval.DAYS or self.when == RolloverInterval.MIDNIGHT:
			self.interval = ONE_DAY_IN_SECONDS  # one day
			self.suffix = "%Y-%m-%d"
			ext_match = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
		elif self.when.startswith("W"):
			if self.when not in RolloverInterval.WEEK_DAYS:
				raise ValueError(f"Invalid day specified for weekly rollover: {self.when}")
			self.interval = ONE_DAY_IN_SECONDS * 7  # one week
			self.day_of_week = int(self.when[1])
			self.suffix = "%Y-%m-%d"
			ext_match = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
		else:
			raise ValueError(f"Invalid RolloverInterval specified: {self.when}")

		self.ext_match = re.compile(ext_match, re.ASCII)
		self.interval = self.interval * interval  # 乘以要求的单位
		# 添加以下行是因为传入的文件名可能是
		# 路径对象（参见问题#27493），但 self.baseFilename 将是一个字符串
		filename = self.absolute_file_path
		if os.path.exists(filename):  # todo: IO. Remove or postpone
			t = int(os.stat(filename).st_mtime)
		else:
			t = int(time.time())
		self.rollover_at = self.compute_rollover(t)

	def compute_rollover(self, current_time: int) -> int:
		"""
		根据指定时间计算出转仓时间。如果在午夜或每周滚动，那么间隔时间是已知的
		需要弄清楚下一个间隔是什么时候
		换句话说，如果在午夜滚动，那么基本间隔是1天，但希望在午夜而不是现在开始这一天的时钟
		因此，必须伪造rollover_at值，以便在正确的时间触发第一次翻转
		之后，定期的休息就会处理好剩下的事情

		注意，不关心闰秒
		"""
		result = current_time + self.interval
		if self.when == RolloverInterval.MIDNIGHT or self.when in RolloverInterval.WEEK_DAYS:
			if self.utc:
				t = time.gmtime(current_time)
			else:
				t = time.localtime(current_time)
			current_hour = t[3]
			current_minute = t[4]
			current_second = t[5]
			current_day = t[6]
			# r是从现在到下一次旋转之间剩余的秒数
			# 计算下一次旋转的日期
			if not self.at_time:
				rotate_ts = ONE_DAY_IN_SECONDS
			else:
				rotate_ts = (self.at_time.hour * 60 + self.at_time.minute) * 60 + self.at_time.second
			r = rotate_ts - ((current_hour * 60 + current_minute) * 60 + current_second)
			if r < 0:
				# 旋转时间在当前时间之前(例如当self.rotateAt是13:45而现在是14:15)，旋转是明天
				r += ONE_DAY_IN_SECONDS
				current_day = (current_day + 1) % 7
			result = current_time + r
			# 如果在某一天进行滚动，请添加到下一次滚动的天数，但偏移1，因为只是计算了到第二天开始的时间
			# 分为三种情况：
			# 情况1: 展期日是今天，在这种情况下，不执行任何操作
			# 情况2: 展期日期在间隔中更远的位置(即，今天是第2天(星期三)，展期是在第6天(星期日)。距下一次展期的天数只是6 - 2 - 1，或3
			# 情况3: 展期日在时间间隔内落后(即，今天是第5天(星期六)，展期日为第3天(星期四)。展期天数为6 - 5 + 3，或4

			# 在这种情况下，它是当前周剩余天数(1)加上下周到滚动日(3)的天数。上面2)和3)中描述的计算需要添加一天。
			# 这是因为上面的时间计算需要我们到这一天的午夜，即第二天的开始
			if self.when in RolloverInterval.WEEK_DAYS:
				day = current_day  # 0 is Monday
				if day != self.day_of_week:
					if day < self.day_of_week:
						days_to_wait = self.day_of_week - day
					else:
						days_to_wait = 6 - day + self.day_of_week + 1
					new_rollover_at = result + (days_to_wait * ONE_DAY_IN_SECONDS)
					if not self.utc:
						dst_now = t[-1]
						dst_at_rollover = time.localtime(new_rollover_at)[-1]
						if dst_now != dst_at_rollover:
							if not dst_now:
								# 夏令时在下一次延期之前生效，因此需要扣除一个小时
								new_rollover_at -= ONE_HOUR_IN_SECONDS
							else:
								# 夏令时在下一次过渡之前结束，因此需要增加一个小时
								new_rollover_at += ONE_HOUR_IN_SECONDS
					result = new_rollover_at
		return result

	def should_rollover(self, record: LogRecord) -> bool:
		"""
		确定是否应发生翻转。未使用记录，因为只是比较时间，但需要它，因此方法签名是相同的
		"""
		t = int(time.time())
		return True if t >= self.rollover_at else False

	async def get_files_to_delete(self) -> List[str]:
		"""
		返回要删除的文件列表
		"""
		dir_name, base_name = os.path.split(self.absolute_file_path)
		loop = get_running_loop()
		file_names = await loop.run_in_executor(None, lambda: os.listdir(dir_name))
		result = []
		prefix = base_name + "."
		p_len = len(prefix)
		for file_name in file_names:
			if file_name[:p_len] == prefix:
				suffix = file_name[p_len:]
				if self.ext_match.match(suffix):
					result.append(os.path.join(dir_name, file_name))
		if len(result) <= self.backup_count:
			return []
		else:
			return result[: len(result) - self.backup_count]

	@staticmethod
	async def _delete_files(file_paths: List[str]):
		"""
		删除文件
		"""
		loop = get_running_loop()
		for file_path in file_paths:
			await loop.run_in_executor(None, lambda: os.remove(file_path))

	async def do_rollover(self):
		"""
		执进行翻转；在这种情况下，当翻转发生时，日期时间戳记会附加到文件名中
		但是，希望文件以间隔的开始时间命名，而不是当前时间
		如果有备份计数，那么必须获取匹配文件名的列表，对它们进行排序并删除具有最旧后缀的文件名
		"""
		if self.stream:
			await self.stream.close()
			self.stream = None
		# 获取该序列开始的时间并将其设为TimeTuple
		current_time = int(time.time())
		dst_now = time.localtime(current_time)[-1]
		t = self.rollover_at - self.interval
		if self.utc:
			time_tuple = time.gmtime(t)
		else:
			time_tuple = time.localtime(t)
			dst_then = time_tuple[-1]
			if dst_now != dst_then:
				if dst_now:
					addend = ONE_HOUR_IN_SECONDS
				else:
					addend = -ONE_HOUR_IN_SECONDS
				time_tuple = time.localtime(t + addend)
		destination_file_path = self.rotation_filename(self.absolute_file_path + "." + time.strftime(self.suffix, time_tuple))
		loop = get_running_loop()
		if await loop.run_in_executor(None, lambda: os.path.exists(destination_file_path)):
			await loop.run_in_executor(None, lambda: os.unlink(destination_file_path))
		await self.rotate(self.absolute_file_path, destination_file_path)
		if self.backup_count > 0:
			files_to_delete = await self.get_files_to_delete()
			if files_to_delete:
				await self._delete_files(files_to_delete)
		await self._init_writer()
		new_rollover_at = self.compute_rollover(current_time)
		while new_rollover_at <= current_time:
			new_rollover_at = new_rollover_at + self.interval
		# 如果 DST 发生变化并且午夜或每周翻转，可对此进行调整
		if self.when == RolloverInterval.MIDNIGHT or self.when in RolloverInterval.WEEK_DAYS and not self.utc:
			dst_at_rollover = time.localtime(new_rollover_at)[-1]
			if dst_now != dst_at_rollover:
				if not dst_now:
					# 夏令时在下一次延期之前生效，因此需要扣除一个小时
					addend = -ONE_HOUR_IN_SECONDS
				else:
					# 夏令时在下一次过渡之前结束，因此需要增加一个小时
					addend = ONE_HOUR_IN_SECONDS
				new_rollover_at += addend
		self.rollover_at = new_rollover_at


class BaseLogHandler(metaclass=SingleTon):
	"""
	基础日志处理类
	"""

	__FILES_HANDLE = {}

	def __init__(self, server_name):
		"""
		:param server_name: 服务名称
		"""
		self.format = LOG_FORMAT
		self.server_name = server_name
		self.log_debug = LOG_DEBUG
		self.__logger: Optional[Logger] = None

	async def get_logger(self, name: str = "log", when=None, count=0, level=logging.DEBUG) -> Logger:
		"""
		异步日志，按日期切割
		:param name: 日志名称
		:param when: 切割时间间隔
		:param count: 保留日志数量
		:param level: 日志级别
		:return: Logger
		"""
		when = when or RolloverInterval.DAYS
		backup_count = count or 3
		logger = self.__FILES_HANDLE.get(name)
		if not logger:
			logger: Logger = Logger(name=name, level=level)
			logfile_path = Path(OUTPUT_PATH, name, f"{name}.log")
			logfile_path.parent.mkdir(parents=True, exist_ok=True)
			logfile_path.touch(exist_ok=True)
			atr_file_handler = AsyncTimedRotatingFileHandler(
				filename=str(logfile_path),
				when=when,
				backup_count=backup_count,
				encoding="utf8"
			)
			formatter = Formatter(self.format)
			atr_file_handler.formatter = formatter
			logger.add_handler(atr_file_handler)
			self.__FILES_HANDLE[name] = logger
		return logger

	async def logger(self):
		"""
		文件日志
		"""
		if not isinstance(self.__logger, Logger):
			self.__logger = await self.get_logger(self.server_name)
		return self.__logger

	async def fail_log(self, msg: str, **kwargs):
		"""
		失败文件
		:param msg: 失败信息
		:param kwargs: 文件夹名称
		"""
		name = kwargs.get("folder_name") or "failed_info"
		log_fail: Logger = await self.get_logger(name)
		await log_fail.info(msg)

	async def error_log(self, data):
		"""
		错误日志
		:param data: 错误信息
		"""
		name = f"{self.server_name}_error"
		await self.fail_log(data, folder_name=name)

	async def info_log(self, *msg):
		"""
		日志信息
		:param msg: 日志信息
		"""
		if self.log_debug:
			print(msg)
		else:
			if not self.__logger:
				await asyncio.create_task(self.logger())  # 此时loop已经在运行了，往该loop中添加协程任务即可
			await self.__logger.info(msg)

	async def log_shutdown(self, name):
		"""
		关闭日志
		"""
		log_sys: Logger = await self.get_logger(name)
		await log_sys.shutdown()