# -*- coding: utf-8 -*-

import logging.config

from rlcards.games.aichess.rlzero.config import CONFIG

"""
配置日志文件和日志级别:
	1.CRITICAL: 50
	2.ERROR: 40
	3.WARNING: 30
	4.INFO: 20
	5.DEBUG: 10
	6:NOTSET: 0
"""

config = {
	'version': 1,
	'formatters': {
		'simple': {
			'format': '%(asctime)s~%(levelname)s:%(levelno)s~%(filename)s:%(lineno)d~%(message)s',
			'datefmt': '%Y-%m-%d %H:%M:%S'
		},
	},
	'handlers': {
		'console': {
			'class': 'logging.StreamHandler',
			'level': 'ERROR',
			'formatter': 'simple'
		},
		'error_file': {
			'class': 'logging.handlers.RotatingFileHandler',
			'filename': CONFIG['error_log_path'],
			'level': 'DEBUG',
			'formatter': 'simple',
			'mode': 'w',
			'maxBytes': 10 ** 7,
			'backupCount': 3,
		},
		'debug_file': {
			'class': 'logging.handlers.RotatingFileHandler',
			'filename': CONFIG['debug_log_path'],
			'level': 'DEBUG',
			'formatter': 'simple',
			'mode': 'w',
			'maxBytes': 10 ** 7,
			'backupCount': 3,
		},
		'operation_file': {
			'class': 'logging.handlers.RotatingFileHandler',
			'filename': CONFIG['operation_log_path'],
			'level': 'DEBUG',
			'formatter': 'simple',
			'mode': 'w',
			'maxBytes': 10 ** 7,
			'backupCount': 3,
		},
	},
	'loggers': {
		'error': {
			'handlers': ['console', 'error_file'],
			'level': 'ERROR',
		},
		'debug': {
			'handlers': ['debug_file'],
			'level': 'DEBUG',
		},
		'simple': {
			'handlers': ['console'],
			'level': 'WARN',
		},
		'operation': {
			'handlers': ['operation_file'],
			'level': 'DEBUG',
		}
	}
}


class HandleConfLog:
	"""
	日志
	"""
	def __init__(self):
		"""
		初始化日志参数
		"""
		logging.config.dictConfig(config)
		self.logger = logging.getLogger()
		# 设置日志输出等级
		self.logger.setLevel(logging.WARNING)

	def error_log(self):
		"""
		错误(error)日志
		"""
		self.logger = logging.getLogger("error")
		return self.logger

	def debug_log(self):
		"""
		调试(debug)日志
		"""
		self.logger = logging.getLogger('debug')
		return self.logger

	def operation_log(self):
		"""
		操作(operate)日志
		"""
		self.logger = logging.getLogger("operation")
		return self.logger

# 实例化日志类
error_log = HandleConfLog().error_log()
debug_log = HandleConfLog().debug_log()
operation_log = HandleConfLog().operation_log()


if __name__=='__main__':
	error_log.error('config_error_level')
	debug_log.debug('config_debug_level')