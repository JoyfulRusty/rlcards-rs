# -*- coding: utf-8 -*-

"""
斗地主休闲场-机器人服务
"""
from public.run.init import start
from base.base_const import ServiceEnum
from predict.ddz.handler import LandlordsHandler


if __name__ == '__main__':
    start(LandlordsHandler, ServiceEnum.ROBOT_LANDLORDS, __file__[0:-3])