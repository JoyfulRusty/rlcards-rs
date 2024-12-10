# -*- coding: utf-8 -*-

"""
打妖怪游戏-机器人服务
"""
from public.run.init import start
from base.base_const import ServiceEnum
from predict.monster.handler import MonsterHandler

print(__file__)
if __name__ == '__main__':
    start(MonsterHandler, ServiceEnum.ROBOT_MONSTER, __file__[0:-3])