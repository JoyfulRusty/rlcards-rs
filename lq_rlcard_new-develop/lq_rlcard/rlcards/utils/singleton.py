# -*- coding: utf-8 -*-


class Singleton(type):
	"""
	创建单列模式
	"""
	# 实例化字典
	_instances = {}

	def __call__(cls, *args, **kwargs):
		"""
		回调
		"""
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

		return cls._instances[cls]