# -*- coding: utf-8 -*-

import threading

class SingleTon(type):
	"""
	单例模式
	"""

	__instances = {}
	__ins_lock = threading.Lock()

	def __call__(cls, *args, **kwargs):
		"""
		当基础该元类的子类的被实例化时调用
		"""
		if cls not in cls.__instances:
			with SingleTon.__ins_lock:
				cls.__instances[cls] = super(SingleTon, cls).__call__(*args, **kwargs)
		return cls.__instances[cls]

class SingletonNew:
	_instances = {}
	_ins_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		if cls not in cls._instances:
			with cls._ins_lock:
				cls._instances[cls] = super(SingletonNew, cls).__new__(cls)
		return cls._instances[cls]

class NoInstances(type):
	"""
	被管理类不能实例化(一般用于只能调用这个类的静态方法)
	"""

	def __call__(cls, *args, **kwargs):
		raise TypeError(f"Can't instantiate {cls.__name__}")

def with_meta(meta, base_class=object):
	"""
	meta: 元类, type或type的派生类, 用于创建类
	等价于继承 base_class, metaclass=meta
	example:
		```
		class Test(with_meta(type, superclass)):
			pass
		```
	"""
	return meta("NewBase", (base_class,), {})