# -*- coding: utf-8 -*-

import asyncio
from utils.init import start
from rlcards.predict.mahjong.handler import MahjongHandler


if __name__ == '__main__':
	# 麻将启动文件
	asyncio.run(start(MahjongHandler, __file__[0:-3]))