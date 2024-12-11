# -*- coding: utf-8 -*-

import redis
import pickle

from rlcards.games.alphazero.aichess import CONFIG

def get_redis_cli():
    """
    获取redis连接
    """
    redis_connect = redis.StrictRedis(
        host=CONFIG['redis_host'],
        port=CONFIG['redis_port'],
        db=CONFIG['redis_db']
    )
    return redis_connect

def get_list_range(redis_cli,name,l,r=-1):
    """
    从redis中获取数据
    """
    assert isinstance(redis_cli,redis.Redis)
    list = redis_cli.lrange(name,l,r)

    return [pickle.loads(d) for d in list]

if __name__ == '__main__':
    redis_connect = get_redis_cli()
    with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
        data_file = pickle.load(data_dict)
        data_buffer = data_file['data_buffer']
    for d in data_buffer:
        redis_connect.rpush('train_data_buffer',pickle.dumps(d))
