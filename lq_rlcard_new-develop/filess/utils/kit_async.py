# -*- coding: utf-8 -*-

import time
import asyncio


class Delay:
    """延迟对象
    """

    def __init__(self, f, *args, **kw):
        self.f = f
        self.args = args
        self.kw = kw

    def call(self):
        """ 返回一个协程，所以外部可以用await等待 """
        if self.f:
            return self.f(*self.args, **self.kw)


class DelayCall:
    """以一个微线程的方式实现一个延时调用
    接受普通函数或者协程函数
    example:
    def p(x):
        print x
    d = DelayCall(5, p, "xx")
    d.start()
    """
    __slots__ = (
        "seconds",
        "delay",
        "task",
        "__start_seconds"
    )

    def __init__(self, seconds, f, *args, **kw):
        assert seconds >= 0, "seconds must be greater than or equal to 0"
        self.seconds = seconds
        self.delay = Delay(f, *args, **kw)
        self.task = None
        self.__start_seconds = 0

    def cancel(self):
        """取消延时调用
        """
        self.task.cancel()

    def left_seconds(self):
        """ 获取计时器的剩余时间 """
        if not self.__start_seconds:
            return self.seconds
        return max(0, self.seconds - (time.time() - self.__start_seconds))

    async def delay_call(self):
        await asyncio.sleep(self.seconds)
        if asyncio.iscoroutinefunction(self.delay.f):
            return await self.delay.call()
        return asyncio.to_thread(self.delay.call())
        # return asyncio.get_running_loop().call_later(self.seconds, self.delay.call)

    def start(self):
        """ 创建一个task """
        self.__start_seconds = time.time()
        self.task = asyncio.create_task(self.delay_call())
        return self.task


class LoopingCall:
    """
    以一个微线程的方式实现一个定时调用 example:
    async def p(x):
        print x
    lc = LoopingCall(5, p, "xx")
    lc.start() # 事件循环中创建一个task
    lc.cancel()
    """

    def __init__(self, seconds, f, *args, **kw):
        assert seconds >= 0, "seconds must be greater than or equal to 0"
        self.seconds = seconds
        self.delay = None
        self.task = None
        self.set_delay_call(f, *args, **kw)

    def set_delay_call(self, f, *args, **kw):
        """ f必须是协程函数 """
        if not asyncio.iscoroutinefunction(f):
            raise ValueError("a coroutine was expected, got {!r}".format(f))
        self.delay = Delay(f, *args, **kw)

    def cancel(self):
        """取消定时调用
        """
        self.task.cancel()

    def start(self):
        """
        start本身是一个协程，这里立即返回，所以外面的await不存在等待，且在内部又创建了任务
        所以此处单纯的创建了一个任务到当前loop中
        """
        self.task = asyncio.create_task(self.loop_call())
        # return await self.task  # 这样就会等待task返回才会继续往下
        return self.task

    async def loop_call(self):
        """ 循环调用 """
        while 1:
            await asyncio.sleep(self.seconds)
            await self.delay.call()