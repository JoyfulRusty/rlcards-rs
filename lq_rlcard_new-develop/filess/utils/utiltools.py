# -*- coding: utf-8 -*-

import re
import math
import hmac
import uuid
import ujson
import orjson
import random
import aiofiles
import dataclasses
import numpy as np

from typing import AnyStr
from datetime import datetime, date, time

from public.const import CONF_PATH

def orjson_obj_to_str(obj):
	"""
	uuid转换为字符串
	"""
	if isinstance(obj, datetime):
		return obj.strftime("%Y-%m-%d %H:%M:%S")
	elif isinstance(obj, date):
		return obj.strftime("%Y-%m-%d")
	elif isinstance(obj, time):
		return obj.strftime("%H:%M:%S")
	if isinstance(obj, uuid.UUID):
		return str(obj)
	elif isinstance(obj, np.ndarray):
		return str(np.array2string(obj, precision=6, floatmode='fixed'))
	elif dataclasses.is_dataclass(obj):
		return str(obj)
	return obj

def orjson_dumps(data):
	"""
	编码文件为bytes
	orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
	"""
	return orjson.dumps(orjson_obj_to_str(data), option=orjson.OPT_SORT_KEYS)

def orjson_loads(data):
	"""
	解码文件为dict
	orjson.loads(data)
	"""
	return orjson.loads(data)

def subtract_by_minutes(curr_time, last_time):
	"""
	计算两个时间之间的分钟差
	"""
	time_delta = datetime.fromtimestamp(curr_time) - datetime.fromtimestamp(last_time)
	return time_delta.seconds / 60

def get_datetime_fmt():
	"""
	获取时间戳年/月/日
	'2024-04-18 14:26:56'
	"""
	return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_date_time():
	"""
	获取年月日时分秒
	datetime.datetime(2024, 4, 18, 14, 4, 22, 160007)
	"""
	return datetime.now()

def get_hour_and_minute_timestamp(timestamp):
	"""
	解析时间戳为具体时间段(小时/分钟)
	"""
	if timestamp:
		dt = datetime.fromtimestamp(timestamp)
		return int(dt.strftime("%H")), int(dt.strftime("%M"))
	return 0, 0

def get_year(date_time=None):
	"""
	取出具体时间中的年份
	date_time == datetime.now()
	"""
	if not date_time:
		date_time = datetime.now()
	return int(date_time.strftime("%Y"))

def get_month(date_time=None):
	"""
	取出具体时间中的月份
	date_time == datetime.now()
	"""
	if not date_time:
		date_time = datetime.now()
	return int(date_time.strftime("%m"))

def get_day(date_time=None):
	"""
	取出具体时间中的天
	date_time == datetime.now()
	"""
	if not date_time:
		date_time = datetime.now()
	return int(date_time.strftime("%d"))

def get_hour(date_time=None):
	"""
	取出具体时间中的小时
	date_time == datetime.now()
	"""
	if not date_time:
		date_time = datetime.now()
	return int(date_time.strftime("%H"))

def get_minute(date_time=None):
	"""
	取出具体时间中的分钟
	date_time == datetime.now()
	"""
	if not date_time:
		date_time = datetime.now()
	return int(date_time.strftime("%M"))

def get_second(date_time=None):
	"""
	取出具体时间中的秒数
	date_time == datetime.now()
	"""
	if not date_time:
		date_time = datetime.now()
	return int(date_time.strftime("%S"))

def random_choice_card(card_arr, p=None):
	"""
	有放回抽样
	从数组中随机选择一张卡牌
	"""
	if not p:
		p = []
	if not isinstance(p, list) or not len(card_arr):
		return
	if p:
		extra_num = len(card_arr) - len(p)
		if extra_num > 0:
			p.extend([0 for _ in range(extra_num)])
		elif extra_num < 0:
			for _ in range(-1 * extra_num):
				p.pop()
	if sum(p) != 1:
		p = None
	return np.random.choice(a=card_arr, size=1, replace=True, p=p)[0]

def random_choice_num(num_arr: list, rate_arr=None):
	"""
	抽取随机数
	"""
	if not rate_arr:
		rate_arr = []
	if not num_arr:
		return
	if rate_arr:
		extra_num = len(num_arr) - len(rate_arr)
		extra_num > 0 and rate_arr.extend([0 for _ in range(extra_num)])
		if extra_num < 0:
			for _ in range(-1 * extra_num):
				rate_arr.pop()
	if sum(rate_arr) != 1:
		rate_arr = None
	return np.random.choice(a=num_arr, size=1, replace=True, p=rate_arr)[0]

def select_element_by_prob(prob: dict, size=100):
	"""
	按概率选择元素，size为转入的概率(必须等于100)
	:param prob: 概率字典 -> {"element1": 10, "element2": 90}
	:param size: 传入概率总和 -> 100
	return: 选择到的元素 -> element2
	"""
	if sum(prob.values()) != size:
		raise ValueError("请检查概率配比")
	x = random.randint(0, size)
	init_prob = 0
	default_item = None
	sort_elements = sorted(prob.items(), key=lambda e: e[1])
	for item, item_pro in sort_elements:
		init_prob += item_pro
		if init_prob >= x:
			default_item = item
			break
	return default_item or max(prob, key=prob.get)

def filter_emoji(input_str, replace='*'):
	"""
	过滤表情
	replace: 未匹配到的替换为
	滤掉非中文、中文字符、英文大小写、数字和常规符号之外的字符
	"""
	filtered_str = re.sub(r"[^\u4e00-\u9fa5\w.。,;:'\"?？!！()\-+=@#&<>/]", replace, input_str)
	return filtered_str

def to_py(data: AnyStr, default=None, log_fun=None):
	if isinstance(data, (dict, list, tuple)):
		return data
	try:
		info = orjson.loads(data)
		return info
	except (SyntaxError, ValueError, TypeError):
		err_info = f'解析数据出错, 原数据:{data}, {type(data)}'
		log_fun(err_info) if callable(log_fun) else print(err_info)
	return default

def to_json(item, _type=1) -> AnyStr:
	"""
	转换为Json字符串
	`ensure_ascii` 是 一个可选参数，用于控制是否将非 ASCII 字符转义为 ASCII 码。
	_type: 1使用ujson, 2使用orjson
	"""
	if isinstance(item, bytes):
		item = item.decode('utf-8')
	try:
		if _type == 1:
			return orjson.dumps(item)
		return ujson.dumps(item, ensure_ascii=False)
	except (SyntaxError, ValueError, TypeError):
		return ""

def packet_command(service_type, cmd, digit=3):
	"""
	打包command命令
	:param service_type:服务类型
	:param cmd:指令
	:param digit:表示 服务类型 与 指令 可以支持的位数
	:return:通过服务类型对cmd命令进行打包传送
	"""
	assert service_type > 0 and cmd >= 0
	num = 10 ** digit
	return service_type * num + cmd

def explode_command(command, digit=3):
	"""
	解析command命令
	:param command: 指令
	:param digit: 表示 服务类型 与 指令 可以支持的位数
	:return: 返回解析结果
	"""
	num = 10 ** digit
	cmd = int(command)
	return cmd // num, cmd % num

def get_hash_secrets(key: str, msg: str, extra_str: str = None, secrets_type="md5"):
	"""
	获取hash加密值,默认需要密钥和消息作为输入
	用于计算消息 身份 验证代码。
	:param key: 密钥
	:param msg: 主体数据
	:param extra_str:额外数据
	:param secrets_type:默认md5
		支持的哈希算法包括：md5、sha1、sha224、sha256、sha384和`sha512`。
	:return:
	"""
	key = key.encode("utf-8")
	md5 = hmac.new(key, msg.encode("utf-8"), digestmod=secrets_type)
	extra_str and md5.update(str(extra_str).encode("utf-8"))
	return md5.hexdigest()


def int_to_bytes(num: int) -> (bytes, int):
	"""
	ASCII 使用一个字节（8位）来表示一个字符
	(num.bit_length() + 7): 获取整数 num 的位数
	为什么要加上 7：当除以 8 时，除法的结果会向下取整到最接近的整数。在位数上加上7 是为了确保计算能够向上取整到最接近的 8 的倍数
	主要是为了保证能够得到足够容纳整数二进制表示的字节数。
	digit: 要使用的字节长度
	"""
	if num < 1:
		raise OverflowError("can't convert negative int to unsigned")
	digit = (num.bit_length() + 7) // 8
	num_bytes = num.to_bytes(digit, byteorder='big')
	return num_bytes, digit


def bytes_to_int_by_bytes_len(b_data: bytes) -> int:
	"""
	根据bytes_len从字节中将int转回
	"""
	return int.from_bytes(b_data, byteorder='big')


def num_is_digit(num: int):
	""" 求一个数值有几位，十进制 """
	assert isinstance(num, int) and num > 0
	return int(math.log10(num) + 1)


def parse_msg_by_bytes(msg: bytes, log_fun=None) -> (int, int, bytes):
	""" 解析字节消息 """
	# 长度标识固定用1位
	# 长度标识：标识cmd占用多少字节长度
	try:
		cmd_bytes_len = bytes_to_int_by_bytes_len(msg[:1])
		cmd = bytes_to_int_by_bytes_len(msg[1:1 + cmd_bytes_len])
		msg = msg[1 + cmd_bytes_len:]
		return explode_command(cmd), msg
	except Exception as err:
		log_fun(f'{msg}消息解析出错：{err}') if log_fun else print(f'{msg}消息解析出错：{err}')
		return (0, 0), b''

def pack_msg_by_bytes(c_type: int, c_code: int, msg: bytes) -> bytes:
	""" 解析字节消息 """
	# 长度标识固定用1位
	# 长度标识：标识cmd占用多少字节长度
	cmd = packet_command(c_type, c_code)
	# 1.先获取命令的字节形式、字节长度
	cmd_bytes, use_bytes_len = int_to_bytes(cmd)
	# 2.再获取命令的 字节长度的 字节形式和长度（一般占1位）
	cmd_len_bytes, cmd_use_bytes_len = int_to_bytes(use_bytes_len)
	assert cmd_use_bytes_len == 1
	return cmd_len_bytes + cmd_bytes + msg

def pack_inner_msg(cmd: int, uid: int, msg: bytes) -> bytes:
	"""
	内部消息频道传输
	解析字节消息
	return: cmd字节占用长度 + cmd字节 + uid字节占用长度 + uid字节 + msg字节
	"""
	# 长度标识固定用1位
	# 长度标识：标识cmd占用多少字节长度
	# 1.先获取命令的字节形式、字节长度
	cmd_bytes, use_bytes_len = int_to_bytes(cmd)
	# 2.再获取命令的 字节长度的 字节形式和长度（一般占1位）
	cmd_len_bytes, cmd_len_bytes_len = int_to_bytes(use_bytes_len)
	assert cmd_len_bytes_len == 1

	# 1.先获取uid的字节形式、字节长度
	uid_bytes, uid_bytes_len = int_to_bytes(uid)
	# 2.第二个是第一个的 字节长度的 字节形式和长度（一般占1位）
	uid_len_bytes, uid_len_bytes_len = int_to_bytes(uid_bytes_len)
	assert uid_len_bytes_len == 1

	# cmd字节占用长度 + cmd字节 + uid字节占用长度 + uid字节 + msg字节
	return cmd_len_bytes + cmd_bytes + uid_len_bytes + uid_bytes + msg

def parse_inner_msg(msg: bytes, log_fun=None) -> (int, int, bytes):
	"""
	内部消息频道传输
	解析字节消息
	msg：cmd字节占用长度 + cmd字节 + uid字节占用长度 + uid字节 + msg字节
	"""
	# 长度标识固定用1位
	# 长度标识：标识cmd占用多少字节长度
	try:
		# 解码code
		cmd_bytes_len = bytes_to_int_by_bytes_len(msg[0:1])
		code = bytes_to_int_by_bytes_len(msg[1: 1 + cmd_bytes_len])

		# 解码uid
		start = 1 + cmd_bytes_len
		end = start + 1
		uid_bytes_len = bytes_to_int_by_bytes_len(msg[start:end])  # uid字节长度
		uid = bytes_to_int_by_bytes_len(msg[end:end + uid_bytes_len])
		msg = msg[end + uid_bytes_len:]
		return (code, uid), msg
	except Exception as err:
		log_fun(f'{msg}消息解析出错：{err}') if log_fun else print(f'{msg}消息解析出错：{err}')
		return (0, 0), b''

async def read_conf_file(file_name: str):
	"""
	读取配置文件
	"""
	async with aiofiles.open(file_name, mode='r') as f:
		return await f.read()

async def read_json_file(key: str):
	"""
	根据key读取json文件里的配置
	"""
	# 读取配置文件
	json_str = await read_conf_file(CONF_PATH)  # 配置文件路径
	if json_str:
		json_data = orjson_loads(json_str)
		return json_data.get(key)

async def to_dict(data: bytes | str):
	"""
	解析json数据格式
	"""
	# 判断是否存在数据
	if not data:
		# await log_handle.fail_log(f"{data}-数据为空[to_dict]")
		return {}
	# 判断数据是否为字典或列表或元组
	if isinstance(data, (dict, list, tuple)):
		return data
	# 捕获bytes/str处理异常
	# 判断是否为字节，数据为字节则进行解码
	if isinstance(data, bytes):
		return orjson.loads(data.decode("utf-8"))
	# 判断数据是否为字符串，数据为字符串则转换为python对象
	if isinstance(data, str):
		return orjson.loads(data)
