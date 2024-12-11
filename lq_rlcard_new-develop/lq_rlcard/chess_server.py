# -*- coding: utf-8 -*-

import asyncio
from utils.init import start
from rlcards.predict.cchess.handler import CCHandler


if __name__ == '__main__':
	# 象棋启动文件
	asyncio.run(start(CCHandler, __file__[0:-3]))