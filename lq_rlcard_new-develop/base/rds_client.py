# -*- coding: utf-8 -*-

import asyncio

from typing import Union
from redis.asyncio import Redis, ConnectionPool

from nsanic.libs.rds_locker import RdsLocker
from nsanic.libs.tool import json_encode, json_parse


class RdsClient:
    """
    redis连接对象
    单例模式请用init方法初始化
    """
    CONN_MAP = {}

    def __init__(self, conf: dict, logs=None):
        """
        初始化参数
        :param conf: 连接配置
        :param logs: 日志对象
        """
        self.__ping = conf.pop('timeout') if 'timeout' in conf else 10
        self.__conf = conf
        self.__logs = logs
        self.__pool = None
        self.__rds_conn = None

    @classmethod
    def init(cls, conf: dict, logs=None):
        """
        单例模型请用该方法进行初始化
        :param conf: 连接配置
        :param logs: 日志对象
        """
        # 判断host/port/db是否存在
        if not all(key in conf for key in ['host', 'port', 'db']):
            raise Exception('conf must contain host, port, db')
        conf_name = f"{conf['host']}-{conf['port']}-{conf['db']}"
        clt = cls.CONN_MAP.get(conf_name)
        if not clt:
            clt = cls(conf, logs=logs)
            cls.CONN_MAP[conf_name] = clt
        return clt

    async def init_rds_pool(self):
        """
        初始化数据库连接池
        """
        try:
            if not self.__rds_conn:
                self.__pool: ConnectionPool = ConnectionPool(**self.__conf)
                self.__rds_conn: Redis = Redis(connection_pool=self.__pool)
            if await self.__rds_conn.ping():
                await self.__logs.info_log(f"redis init connection pool success: ********")
        except Exception as err:
            await self.__logs.error_log(f"redis init connection pool error: {err}")

    def check_rds_loop_ping(self, loop=None):
        """
        检测并设置redis连接池循环
        """
        if not loop:
            loop = asyncio.get_event_loop()
        loop.create_task(self.__rds_loop_ping())

    async def __rds_loop_ping(self):
        """
        ping-redis连接池
        """
        while 1:
            if await self.__rds_conn.ping():
                await self.__logs.info_log(f"redis ping connection pool success: *******")
            else:
                try:
                    await self.init_rds_pool()
                except Exception as err:
                    await self.__logs.error_log(f"redis ping connection pool has failed, will retry in {self.__ping} seconds: {err}")
                    self.__pool = None
            await asyncio.sleep(self.__ping)

    @property
    def __conn(self):
        """
        获取redis连接对象
        """
        if not self.__rds_conn:
            raise Exception("can not found redis connection pool")
        return self.__rds_conn

    @property
    def conn(self):
        """
        获取redis连接对象
        """
        return self.__conn

    async def locked(self, key: str, fun=None, fun_param: (list, tuple) = None, time_out=5, pre_key='LOCKER__'):
        """
        锁定执行
        :param key: 锁定标识
        :param fun: 执行函数
        :param fun_param: 函数参数
        :param time_out: 自动释放时间
        :param pre_key: 锁定标识预制前缀，用于区分锁定标识采用redis key本身
        """
        try:
            if not fun_param:
                fun_param = ()
            if pre_key:
                key = f'{pre_key}{key}'
            async with self.__conn.lock(key, lock_class=RdsLocker, timeout=time_out):
                return callable(fun) and ((await fun(*fun_param)) if asyncio.iscoroutinefunction(fun) else fun(*fun_param))
        except Exception as err:
            await self.__logs.error_log(f'redis locked has failed: {err}')

    async def get_item(self, key: Union[str, bytes], is_json=False):
        """
        获取缓存
        :param key: 缓存名称
        :param is_json: 是否需要解析成JSON
        """
        try:
            if await self.__conn.exists(key):
                info = await self.__conn.get(key)
                return json_parse(info) if is_json else info
            return
        except Exception as err:
            await self.__logs.error_log(f'redis get has failed: {err}')

    async def del_item(self, *args):
        """
        删除缓存
        :param args: 缓存名称列表
        """
        try:
            if not args:
                return
            return await self.__conn.delete(*args)
        except Exception as err:
            await self.__logs.error_log(f'redis del has failed: {err}')

    async def set_item(
            self, key: Union[str, bytes], value: Union[str, int, float, bytes, list, tuple, dict], ex_time: int = None):
        """
        设置字符串缓存
        :param key: 缓存名称
        :param value: 缓存值，非字符串类型将会自动转换为字符串存储
        :param ex_time: 缓存有效期，默认不设置，既永久有效，类型数字类型 单位：秒 # type:int
        """
        try:
            if isinstance(value, (list, tuple, dict)):
                value = json_encode(value)
            return await self.__conn.set(key, value, ex=ex_time)
        except Exception as err:
            await self.__logs.error_log(f'redis set has failed: {err}')

    async def exists(self, key: Union[str, bytes]):
        """
        检查key是否存在
        :param key: 缓存名称
        """
        try:
            return await self.__conn.exists(key)
        except Exception as err:
            await self.__logs.error_log(f'redis exists has failed: {err}')

    async def drop_item(self, key: Union[str, bytes]):
        """
        根据key删除某项缓存
        :param key: 缓存名称
        """
        try:
            return await self.__conn.delete(key)
        except Exception as err:
            await self.__logs.error_log(f'redis drop has failed: {err}')

    async def get_hash(self, key: Union[str, bytes], h_key: Union[str, bytes], is_json=False):
        """
        获取Hash值

        :param key: 缓存名称
        :param h_key: 缓存键名称
        :param is_json: 是否为JSON
        """
        try:
            if await self.__conn.hexists(key, h_key):
                info = await self.__conn.hget(key, h_key)
                return json_parse(info) if is_json else info
        except Exception as err:
            await self.__logs.error_log(f'redis get_hash has failed: {err}')

    async def set_hash(self, key: Union[str, bytes], h_key: Union[str, bytes], data):
        """
        设置Hash

        :param key: 缓存名称
        :param h_key: 缓存键名称
        :param data: 保存的数据
        """
        try:
            if isinstance(data, list) or isinstance(data, dict):
                data = json_encode(data, log_fun=self.__logs.error if self.__logs else None)
            return await self.__conn.hset(key, h_key, data)
        except Exception as err:
            await self.__logs.error_log(f'redis set_hash has failed: {err}')

    async def drop_hash(self, key: Union[str, bytes], h_key: Union[str, bytes]):
        """
        删除Hash值
        :param key: 缓存名称
        :param h_key: 缓存键名称
        """
        try:
            return await self.__conn.hdel(key, h_key)
        except Exception as err:
            await self.__logs.error_log(f'redis drop_hash has failed: {err}')

    async def drop_hash_bulk(self, key: Union[str, bytes], key_list: list or tuple):
        """
        批量删除Hash值
        :param key: 缓存名称
        :param key_list: 缓存键列表
        """
        try:
            key_list and (await self.__conn.hdel(key, *key_list))
        except Exception as err:
            await self.__logs.error_log(f'redis drop_hash_bulk has failed: {err}')

    async def get_hash_all(self, key: Union[str, bytes]):
        """
        获取整个Hash
        :param key: 缓存名
        """
        try:
            return await self.__conn.hgetall(key)
        except Exception as err:
            await self.__logs.error_log(f'redis get_hash_all has failed: {err}')

    async def get_hash_val(self, key: Union[str, bytes]):
        """
        获取整个Hash所有值
        :param key: 缓存名
        """
        try:
            return await self.__conn.hvals(key)
        except Exception as err:
            await self.__logs.error_log(f'redis get_hash_val has failed: {err}')

    async def set_hash_bulk(self, key: Union[str, bytes], h_map: dict):
        """
        通过字典映射设置或更新多个Hash
        :param key: 缓存名
        :param h_map: 字典映射 -- 必须是{str: str} 或{bytes: str} 或{bytes: bytes} 几种形式
        """
        try:
            return await self.__conn.hset(key, mapping=h_map)
        except Exception as err:
            await self.__logs.error_log(f'redis set_hash_bulk has failed: {err}')

    async def scan_hash(self, key: Union[str, bytes], start: int = 0, count: int = 10, match: str = None):
        """
        按匹配的条件迭代扫描hash
        """
        try:
            return await self.__conn.hscan(key, cursor=start, match=match, count=count)
        except Exception as err:
            await self.__logs.error_log(f'redis scan_hash has failed: {err}')

    async def pub_sub(self, channel: Union[str, list, tuple]):
        """
        消息订阅模型
        """
        sub_item = self.__conn.pubsub()
        if isinstance(channel, str):
            channel = [channel]
        await sub_item.subscribe(*channel)
        return sub_item

    async def publish(self, channel: str, msg: (bytes, str)):
        """
        发布消息
        :param channel: 消息信道
        :param msg: 发布的消息
        """
        return await self.__conn.publish(channel, msg)

    async def qlpush(self, key: str, data_list: list):
        """
        左侧入队列
        :param key: 队列/集合名称
        :param data_list: 入队列表 --type: list
        """
        try:
            values = []
            for item in data_list:
                if isinstance(item, dict) or isinstance(item, list):
                    item = json_encode(item, log_fun=self.__logs.error if self.__logs else None)
                values.append(item)
            return await self.__conn.lpush(key, *values)
        except Exception as err:
            await self.__logs.error_log(f'redis qlpush has failed: {err}')

    async def qrpush(self, key: str, data_list: list):
        """
        右侧入队列
        :param key: 队列/集合名称
        :param data_list: 入队列表 --type: list
        """
        try:
            values = []
            for item in data_list:
                if isinstance(item, dict) or isinstance(item, list):
                    item = json_encode(item, log_fun=self.__logs.error if self.__logs else None)
                values.append(item)
            return await self.__conn.rpush(key, *values)
        except Exception as err:
            await self.__logs.error_log(f'redis qrpush has failed: {err}')

    async def qrpop(self, key: str, count=1):
        """
        右侧出队

        :param key: 队列/集合名称
        :param count: 出队数量
        """
        try:
            return await self.__conn.rpop(key, count=None if count <= 1 else count)
        except Exception as err:
            await self.__logs.error_log(f'redis qrpop has failed: {err}')

    async def qlpop(self, key: str, count=1):
        """
        左侧出队

        :param key: 队列/集合名称
        :param count: 出队数量
        """
        try:
            return await self.__conn.lpop(key, count=None if count <= 1 else count)
        except Exception as err:
            await self.__logs.error_log(f'redis qlpop has failed: {err}')

    async def qlen(self, key: str):
        """
        获取队列长度
        :param key: 队列/集合名称
        """
        try:
            return await self.__conn.llen(key)
        except Exception as err:
            await self.__logs.error_log(f'redis qlen has failed: {err}')

    async def expired(self, key: Union[str, bytes], t: int, nx=False, xx=False, gt=False, lt=False):
        """
        设置过期时间
        :param key: 键
        :param t: 过期时间
        :param nx: 只有键不存在时，才设置值
        :param xx: 只有键存在时，才设置值
        :param gt: 只有值大于给定值时，才设置值
        :param lt: 只有值小于给定值时，才设置值
        :return: 设置成功返回1，否则返回0
        """
        try:
            return await self.__conn.expire(key, time=t, nx=nx, xx=xx, gt=gt, lt=lt)
        except Exception as err:
            await self.__logs.error_log(f'redis expired has failed: {err}')

    async def add_count(self, key: Union[str, bytes]):
        """
        自增计数器
        :param key: 键
        """
        try:
            return await self.__conn.incr(key)
        except Exception as err:
            await self.__logs.error_log(f'redis add_count has failed: {err}')

    async def save_hash_kv(self, name, key, data):
        """
        批量存储哈希：{"a": str(data), "b", str(data)}
        备注：在设置字典map前，需要将值手动转换成字符串或bytes类型
        """
        try:
            await self.__conn.hset(name, key, data)
        except Exception as err:
            await self.__logs.error_log(f'redis save_hash_kv has failed: {err}')

    async def delete_hash_kv(self, name, key):
        """
        删除哈希键值对
        :param name: 哈希名称
        :param key: 键
        """
        try:
            await self.__conn.hdel(name, key)
        except Exception as err:
            await self.__logs.error_log(f'redis delete_hash_kv has failed: {err}')

    async def get_hash_kv(self, name, key):
        """
        获取哈希键值对
        :param name: 哈希名称
        :param key: 键
        """
        try:
            return await self.__conn.hget(name, key)
        except Exception as err:
            await self.__logs.error_log(f'redis get_hash_kv has failed: {err}')

    async def ordered_set_zadd(self, name, data):
        """
        redis有序集合添加数据
        :param name: 有序集合名称
        :param data: 数据
        """
        try:
            return await self.__conn.zadd(name, data)
        except Exception as err:
            await self.__logs.error_log(f'redis ordered_set_zadd has failed: {err}')

    async def get_ordered_set_zrange(self, name, start=0, end=-1):
        """
        redis有序集合读取数据
        :param name: 有序集合名称
        :param start: 开始位置
        :param end: 结束位置
        """
        try:
            return await self.__conn.zrange(name, start, end, withscores=True)
        except Exception as err:
            await self.__logs.error_log(f'redis get_ordered_set_zrange has failed: {err}')

    async def get_str(self, name):
        """
        读取字符串
        :param name: 字符串名称
        """
        try:
            return await self.__conn.get(name)
        except Exception as err:
            await self.__logs.error_log(f'redis get_str has failed: {err}')

    async def save_str(self, name, value):
        """
        存储字符串
        :param name: 字符串名称
        :param value: 字符串值
        """
        try:
            return await self.__conn.set(name, json_encode(value))
        except Exception as err:
            await self.__logs.error_log(f'redis save_str has failed: {err}')

    async def save_list_push(self, name, data):
        """
        存储list数据
        :param name: list名称
        :param data: 数据
        """
        try:
            await self.__conn.rpush(name, *data)
        except Exception as err:
            await self.__logs.error_log(f'redis save_list_push has failed: {err}')

    async def get_list_pop(self, name):
        """
        读取list数据
        :param name: list名称
        """
        try:
            return await self.__conn.lpop(name)
        except Exception as err:
            await self.__logs.error_log(f'redis get_list_pop has failed: {err}')

    async def get_list_lrange(self, name, start=0, end=-1):
        """
        读取list中的全部数据
        :param name: list名称
        :param start: 开始位置
        :param end: 结束位置
        """
        try:
            return await self.__conn.lrange(name, start, end)
        except Exception as err:
            await self.__logs.error_log(f'redis get_list_lrange has failed: {err}')

    async def ltrim_list(self, name, start=0, end=-1):
        """
        只保留[start, end]条数据
        裁剪list，仅保留[start, end]
        :param name: list名称
        :param start: 开始位置
        :param end: 结束位置
        """
        try:
            return await self.__conn.ltrim(name, start, end)
        except Exception as err:
            await self.__logs.error_log(f'redis ltrim_list has failed: {err}')

    async def get_list_delete(self, name):
        """
        删除list
        :param name: list名称
        """
        try:
            return await self.__conn.lrange(name, 0, -1)
        except Exception as err:
            await self.__logs.error_log(f'redis get_list_delete has failed: {err}')

    async def get_name_len(self, name):
        """
        获取list长度
        :param name: list名称
        """
        try:
            return await self.__conn.llen(name)
        except Exception as err:
            await self.__logs.error_log(f'redis get_name_len has failed: {err}')

    async def add_online_robot(self, uid, name="ONLINE_PLAYER_DATA"):
        """
        添加在线玩家
        :param uid: 玩家ID
        :param name: 存储名称
        """
        try:
            return await self.__conn.sadd(name, *uid)
        except Exception as err:
            await self.__logs.error_log(f'redis add_online_robot has failed: {err}')

    async def delete_online_robot(self, uid, name="ONLINE_PLAYER_DATA"):
        """
        删除在线玩家
        :param uid: 玩家ID
        :param name: 存储名称
        """
        try:
            return await self.__conn.srem(name, *uid)
        except Exception as err:
            await self.__logs.error_log(f'redis delete_online_robot has failed: {err}')

    async def check_online_robot(self, uid, name="ONLINE_PLAYER_DATA"):
        """
        检查在线玩家
        :param uid: 玩家ID
        :param name: 存储名称
        """
        try:
            return 1 if await self.__conn.sismember(name, uid) else 0
        except Exception as err:
            await self.__logs.error_log(f'redis check_online_robot has failed: {err}')