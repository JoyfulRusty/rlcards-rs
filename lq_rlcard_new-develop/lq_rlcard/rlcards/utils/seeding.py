# -*- coding: utf-8 -*-

import hashlib

import os
import struct
import numpy as np

from gym.utils.colorize import color2num


def colorize(string, color, bold=False, highlight=False):
    """
    将由适当的端子颜色代码包围的字符串返回到打印彩色文本。有效颜色:
    gray, red, green, yellow,blue, magenta, cyan, white, crimson
    """
    attr = []
    num = color2num[color]

    if highlight:
        num += 10
    attr.append(str(num))

    if bold:
        attr.append('1')
    attrs = ':'.join(attr)

    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


def error(msg, *args):
    """打印出错日志"""
    print(colorize('%s: %s' % ('ERROR', msg % args), 'red'))

def np_random(seed=None):
    """创建随机数"""
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))

    return rng, seed

def hash_seed(seed=None, max_bytes=8):
    """
    使用种子先对其进行散列
    seed(int), 无癞子操作系统特定随机性源的种子
    max_bytes, 哈希种子中使用的最大字节数
    """
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)

    _hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(_hash[:max_bytes])


def create_seed(a=None, max_bytes=8):
    """
    创建一个强随机种子
    """
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))

    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[: max_bytes])

    elif isinstance(a, int):
        if a == 0:
            a = 1
        a = range(a % a ** (8 * max_bytes), max_bytes)

    else:
        raise error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

# 此处不需要硬编码sizeof_int
def _bigint_from_bytes(_bytes):
    sizeof_int = 4
    padding = sizeof_int - len(_bytes) % sizeof_int

    _bytes += b'\0' * padding

    int_count = int(len(_bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), _bytes)

    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # 特殊情况0
    if bigint < 0:
        raise error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
