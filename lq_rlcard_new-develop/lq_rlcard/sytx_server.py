import asyncio
from utils.init import start
from rlcards.predict.sytx_gz.handler import SyHandler


if __name__ == '__main__':
	# 麻将启动文件
	asyncio.run(start(SyHandler, __file__[0:-3]))