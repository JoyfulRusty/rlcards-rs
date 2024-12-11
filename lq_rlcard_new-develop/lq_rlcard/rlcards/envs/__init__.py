# -*- coding: utf-8 -*-

from rlcards.envs.env import Env
from rlcards.envs.registration import register, make


# TODO: 注册游戏环境(打妖怪，斗地主，麻将，贵阳麻将, 水鱼天下)
register(
    env_id='monster',
    entry_point='rlcards.envs.monster:MonsterEnv',
)

register(
    env_id='doudizhu',
    entry_point='rlcards.envs.ddzhu:DdzEnv',
)

register(
    env_id='mahjong',
    entry_point='rlcards.envs.mahjong:MahjongEnv',
)

register(
    env_id='gymahjong',
    entry_point='rlcards.envs.gymahjong:GyMahjongEnv'
)

register(
    env_id='sytx',
    entry_point='rlcards.envs.sytx:SyEnv'
)

register(
    env_id='sytx_gz',
    entry_point='rlcards.envs.sytx_gz:SyEnv'
)

register(
    env_id='pig',
    entry_point='rlcards.envs.pig:PigEnv'
)
