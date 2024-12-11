import ast

import ujson
import aiofiles

from public.const import CONF_PATH
from public.logs import log_handle

def with_meta(meta, *_class):
    """ 等价于继承 base_class, metaclass=meta """
    return meta("NewBase", _class, {})


async def read_file(filename: str):
    """ 读取json文件 """
    async with aiofiles.open(filename, mode='r') as f:
        json_str = await f.read()
        return json_str


async def read_json_file(key: str) -> dict:
    json_str = await read_file(CONF_PATH)
    if json_str:
        try:
            json_data = ujson.loads(json_str)
            return json_data.get(key)
        except Exception as e:
            await log_handle.fail_log(f"解析JSON配置出错:{e}")


def json_dumps(data):
    """ 序列化数据，将python内置数据转换为json字符串 """
    return ujson.dumps(data)


async def to_dict(data: bytes or str):
    if data:
        if isinstance(data, dict) or isinstance(data, list):
            return data
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        try:
            return ast.literal_eval(data)
        except (ValueError, SyntaxError):
            pass
        if isinstance(data, str):
            try:
                return ujson.loads(data)
            except Exception as e:
                await log_handle.fail_log('字典转换数据格式错误', data, e)
    return {}


def remove_by_value(data, value, remove_count=1):
    """
    :param data: list
    :param value:
    :param remove_count: 为-1的时候表示删除全部, 默认为1
    :return: already_remove_count: int
    """
    data_len = len(data)
    count = remove_count == -1 and data_len or remove_count
    already_remove_count = 0
    for i in range(0, count):
        if value in data:
            data.remove(value)
            already_remove_count += 1
        else:
            break
    return already_remove_count


if __name__ == '__main__':
    import asyncio

    res = asyncio.run(read_json_file("redis_hall"))
    print(res)