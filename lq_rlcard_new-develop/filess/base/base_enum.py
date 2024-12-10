# -*- coding: utf-8 -*-

from enum import IntEnum, Enum


class BaseEnum(IntEnum):
	"""
	封装基础IntEnum枚举
	"""

	def __new__(cls, value, phrase, description=''):
		"""
		构建int枚举类型对象
		"""
		obj = int.__new__(cls, value)
		obj._value_ = value
		obj.val = value
		obj.phrase = phrase  # 原因短语
		obj.desc = description  # 描述
		return obj

	@classmethod
	def find_member_by_val(cls, value):
		"""
		在枚举类中查找具有特定值的成员
		"""
		return cls._value2member_map_.get(value)

	@classmethod
	def map_list(cls):
		"""
		以value--label map的方式返回类型映射
		"""
		all_list = []
		for v in cls._value2member_map_.values():
			all_list.append({'value': v.val, 'label': v.phrase})
		return all_list


class BaseStrEnum(str, Enum):
	"""
	封装基础StrEnum枚举
	python-version(未实现StrEnum) < 3.11(已经实现了StrEnum)
	"""

	def __new__(cls, value):
		"""
		构建str枚举类型对象
		"""
		obj = str.__new__(cls, value)
		obj._value_ = value
		obj.val = value
		return obj

	@classmethod
	def from_string(cls, name, string):
		"""
		转换字符串
		"""
		values = [s.strip() for s in string.split(',')]
		return cls(name, *values)