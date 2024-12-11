import aioredis
from utils.utils import read_json_file
from public.logs import log_handle
import traceback


class BaseRedis:
    """
    基础redis连接
    """
    CONN_DIC = {}  # 连接对象

    def __init__(self, redis_key="redis_hall"):
        """
        初始化参数
        """
        self.__conf_name = redis_key
        self.__conn = self.CONN_DIC.get(redis_key)
        # asyncio.create_task(self.init_conn())

    async def init_conn(self):
        """
        连接redis
        """
        print("初始化redis...")
        if not self.__conn:
            try:
                params = await read_json_file(self.__conf_name)  # 读取config_json文件
                url = params.pop("url")
                self.__conn = aioredis.from_url(url, **params)  # 连接池（内部会自动调用空闲连接）
            except Exception as data:
                print(traceback.format_exc(data))
                await log_handle.fail_log("Redis连接出错，请检查redis的链接配置：", data)
                return
            self.CONN_DIC[self.__conf_name] = self.__conn
        return self.__conn

    async def close(self):
        """
        关闭连接
        """
        conn = self.__conn
        if conn:
            self.__conn = None
            await conn.close()

    async def save_hash_kv(self, name, key, data):
        """
        批量存储哈希：{"a": str(data), "b", str(data)}
        备注：在设置字典map前，需要将值手动转换成字符串或bytes类型
        """
        try:
            await self.__conn.hset(name, key, data)
        except Exception as err:
            await log_handle.fail_log(err)

    async def get_hash(self, name):
        return await self.__conn.hgetall(name)

    async def get_hash_kv(self, name, key):
        return await self.__conn.hget(name, key)

    async def publish(self, channel, msg):
        await self.__conn.publish(channel, msg)

    async def subscribe(self, channel: str):
        obj = self.__conn.pubsub()
        await obj.subscribe(channel)
        return obj

    async def psubscribe(self, channel: str):
        """
        订阅一个或多个符合给定模式的频道。
        每个模式以 * 作为匹配符，比如 it* 匹配所有以 it 开头的频道( it.news 、 it.blog 、 it.tweets 等等)。
        """
        obj = self.__conn.pubsub()
        await obj.psubscribe(channel)
        return obj

    async def ping(self):
        try:
            await self.__conn.ping()
        except Exception as e:
            log_handle.fail_log("redis ping error, ", e)

    async def read_queue(self, name):
        """ 出队 """
        if not self.__conn:
            await self.init_conn()
        # print("name: ", name)
        return await self.__conn.lpop(name)