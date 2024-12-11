import asyncio

from utils.init import start
from rlcards.predict.pig.handler import PigHandler

if __name__ == '__main__':
	# 拱猪启动文件
	asyncio.run(start(PigHandler, __file__[0:-3]))