import asyncio
from rlcards.predict.monster.handler import MonsterHandler
from utils.init import start


if __name__ == '__main__':
    # 打妖怪启动文件
    asyncio.run(start(MonsterHandler, __file__[0:-3]))