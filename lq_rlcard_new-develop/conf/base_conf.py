# -*- coding: utf-8 -*-

from base.rmq_client import RmqClient
from base.rds_client import RdsClient
from base.mysql_client import MysqlClient

from base.base_logger import BaseLogHandler


from utils.meta_class import SingleTon

class BaseConf(metaclass=SingleTon):
	"""
	基础配置
	"""
	SERVER_NAME = "MAIN"  # 服务名称
	SERVER_ID = "M0001"  # 服务ID，建议4-6个字符 不同的服务之间该标识定义必须不一致
	RUN_WORKER = 1  # 工作进程，该项配置可依据CPU核心数配置，最佳值为CPU核心数
	CONF_DB = None
	CONF_RDS = None
	CONF_AMQP = None
	rmq: RmqClient = None
	rds: RdsClient = None
	log: BaseLogHandler = None
	msql: MysqlClient = None

	@classmethod
	def set_conf(cls, server_name, server_id):
		"""
		设置rmq配置
		"""
		cls.SERVER_NAME = server_name
		cls.SERVER_ID = f"{server_name}_{server_id}"
		cls.log = BaseLogHandler(server_name)
		if cls.CONF_RDS:
			cls.rds = RdsClient.init(cls.CONF_RDS['default'], logs=cls.log)
		if cls.CONF_AMQP:
			cls.rmq = RmqClient.init(cls.CONF_AMQP['default'], logs=cls.log)
		if cls.CONF_DB:
			cls.msql = MysqlClient.init(cls.CONF_DB["default"], logs=cls.log)

init_base_conf = BaseConf()