# -*- coding: utf-8 -*-

from base.base_logger import BaseLogHandler
from conf.base_conf import init_base_conf
from base.base_const import RmqGameChannel


class BaseService(BaseLogHandler):
	"""
	基础服务
	"""

	__slots__ = (
		"__listen_channel",
		"__server_id",
		"__service_name",
		"__queue_name",
		"__service_info",
		"__conf",
		"__service_type",
		"__commands"
	)

	def __init__(self, server_name):
		"""
		初始化参数
		:param server_name: 服务器名称
		"""
		super().__init__(server_name)
		self.__conf = init_base_conf  # 配置redis和rabbitmq服务
		self.__service_type = 0
		self.__commands = {}
		self.__listen_channel = []
		self.__server_id = 1  # 服务器ID
		self.__service_name = ""
		self.__queue_name = None
		self.__service_info = None

	@property
	def conf(self):
		"""
		配置rmq and rds (rabbitmq/redis)
		"""
		return self.__conf

	@property
	def service_type(self):
		"""
		服务类型
		"""
		return self.__service_type

	@service_type.setter
	def service_type(self, s_type):
		"""
		服务类型
		:param s_type: 服务类型
		"""
		self.__service_type = s_type

	@property
	def commands(self):
		"""
		命令
		"""
		return self.__commands

	@commands.setter
	def commands(self, mds):
		"""
		回调命令
		:param mds: 命令列表
		"""
		self.__commands = mds

	@property
	def listen_channel(self):
		"""
		监听频道
		"""
		return self.__listen_channel

	@property
	def sid(self):
		"""
		sid
		"""
		return self.server_id

	@sid.setter
	def sid(self, sd):
		"""
		sid
		:param sd: sid
		"""
		self.server_id = sd

	@property
	def server_id(self):
		"""
		服务ID
		"""
		return self.__server_id

	@server_id.setter
	def server_id(self, sid):
		"""
		服务ID
		:param sid: 服务ID
		"""
		self.__server_id = sid

	@property
	def service_name(self):
		"""
		服务名
		"""
		return self.__service_name

	@service_name.setter
	def service_name(self, s_name):
		"""
		服务名
		:param s_name: 服务名
		"""
		self.__service_name = s_name

	@property
	def queue_name(self):
		"""
		队列名
		"""
		return self.__queue_name

	@queue_name.setter
	def queue_name(self, q_name):
		"""
		队列名
		:param q_name: 队列名
		"""
		self.__queue_name = q_name

	def __add_handler(self, cmd, func):
		"""
		添加回调命令
		:param cmd: 方法
		:param func: 回调函数
		"""
		assert cmd and cmd > 0 and func
		assert self.__commands.get(cmd) is None, "duplicate handler"
		self.__commands[cmd] = func

	def add_handlers(self, cmd_handlers):
		"""
		添加回调命令
		:param cmd_handlers: 命令回调字典
		"""
		for k, v in list(cmd_handlers.items()):
			self.__add_handler(k, v)

	async def setup(self, server_id, service_type, service_name=None, server_info=None):
		"""
		注册信息
		:param server_id: 服务ID
		:param service_type: 服务类型
		:param service_name: 服务名
		:param server_info: 服务信息
		"""
		assert server_id > 0
		self.__server_id = server_id
		self.__service_type = service_type
		self.__service_name = (service_name or self.__class__.__name__).lower()
		self.__queue_name = f"{self.__service_name}_{self.service_type}"
		self.__service_info = server_info
		self.__conf.set_conf(self.__service_name, self.__server_id)
		# rabbitmq -> 监听队列信息: 子游戏发送过来的队列名
		self.__listen_channel.append(f'{RmqGameChannel.C_SERVICES}_{service_type}')
		print(
			f"""
			启动服务: {self.__service_name}
			服务号: {self.service_type}
			监听频道：{self.__listen_channel}
			rmq监听频道(交换器)：{self.__listen_channel[0]}
			rmq监听路由键：{self.__queue_name}
			rmq监听队列(该队列绑定了上面的交换器与路由键)：{self.__queue_name}
			server_id: {self.__server_id}
			_service_info: {self.__service_info}
			""")

	async def service(self, cmd, uid, data):
		"""
		执行子游戏中对应的回调计算服务
		:param cmd: 命令
		:param uid: 用户ID
		:param data: 数据
		"""
		func = self.__commands.get(cmd)
		if not func or not callable(func):
			return
		await func(uid, data)