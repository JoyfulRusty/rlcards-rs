# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import argparse
from public.run.daemon import daemonize
from base.base_const import ServiceEnum

# logs->输出路径
OUTPUT_PATH_DAEMON = os.path.join(os.getcwd(), 'output', 'daemon')


def start(server_class, server_type: ServiceEnum, file_name):
    """
    启动服务器
    :param server_class: 服务类
    :param server_type: 服务类型
    :param file_name: 服务文件名
    """
    assert server_class
    assert file_name

    # 初始化参数
    args = init_args()
    # nt -> windows
    if os.name == "nt":
        file_name = file_name.replace("\\", "/")
    file_name = file_name.split("/")[-1]  # win下不支持转换???
    if not args.stop:
        print(f"Start: python3 {file_name}.py {server_class.__name__}")

    # 判断是否退出
    def on_exit(*params):
        """
        是否退出
        """
        asyncio.create_task(server_class.share_server().on_signal_stop())

    # 服务id
    server_id = args.sid or 1
    server_name = server_type.phrase
    if args.stop:
        return stop(server_name, server_id, file_name, server_class)
    if args.d:  # daemon启动
        start_daemon(server_name, server_id, on_exit)
    elif args.restart:  # todo: 此处有问题（暂时别用）
        stop(server_name, server_id, file_name, server_class)
        start_daemon(server_name, server_id, on_exit)
    asyncio.run(_do_start(server_class, server_type, server_name, server_id))


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', action='store_true')
    parser.add_argument('--d', action='store_true', help="后台启动")
    parser.add_argument('--sid', type=int, default=1, help="服务系列号index")
    parser.add_argument('--stop', action='store_true', help="停止服务")
    parser.add_argument('--restart', action='store_true', help="重启")
    args = parser.parse_args()
    if args.d and args.stop:
        raise RuntimeError("--d and --stop can't at the same time")
    if args.d and args.restart:
        raise RuntimeError("--d and --restart can't at the same time")
    if args.stop and args.restart:
        raise RuntimeError("--stop and --restart can't at the same time")
    return args


def start_daemon(server_name, server_id, on_exit):
    """
    后台启动
    :param server_name: 服务名称
    :param server_id: 服务id
    :param on_exit: 退出函数
    """
    # 创建一个守护进程，并将进程ID和核心转储文件保存到指定的输出路径
    try:
        # 判断文件夹是否存在
        if not os.path.exists(OUTPUT_PATH_DAEMON):
            os.makedirs(OUTPUT_PATH_DAEMON)
        pid_file = os.path.join(OUTPUT_PATH_DAEMON, f"{server_name}.pid_{server_id}")
        core_dump_file = os.path.join(OUTPUT_PATH_DAEMON, f"{server_name}.log")
        print("dump_file", core_dump_file, pid_file)
        daemonize(pid_file, stderr=core_dump_file, on_exit=on_exit)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)


def stop(server_name, server_id, file_name, server_class):
    """
    停止服务
    :param server_name: 服务名称
    :param server_id: 服务id
    :param file_name: 文件名
    :param server_class: 服务类
    """
    # 用于停止一个正在运行的Python进程
    # 具体来说，它首先导入signal模块和os模块，然后定义一个pid_file变量，用于存储进程ID（PID）的文件路径。
    # 接着，它检查pid_file是否存在，
    # 如果存在，就从文件中读取进程ID，并打印一条消息，表示要停止的进程的名称、PID和文件路径。然
    # 后，使用os.kill()函数向该进程发送SIGTERM信号，以请求其正常退出。
    # 如果pid_file不存在，就打印一条错误消息，并引发一个SystemExit异常，以终止程序的执行。
    import signal
    pid_file = os.path.join(OUTPUT_PATH_DAEMON, f"{server_name}.pid_{server_id}")
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            process_id = int(f.read())
            print(f"Stop: python3 {file_name}.py {server_class.__name__}, {pid_file}, {process_id}")
            os.kill(process_id, signal.SIGTERM)
    else:
        print('Not running', file=sys.stderr)
        raise SystemExit(1)


async def _do_start(server_class, server_type, server_name, server_id):
    """
    启动服务
    返回一个共享的服务器实例，然后调用该实例的setup()方法
    该方法接受三个参数：server_id(服务器ID)、server_type(服务器类型)和server_name(服务器名称)
    """
    # setup()方法用于设置服务器的配置信息
    await server_class.share_server(server_name).setup(server_id, server_type, server_name)

    # 调用共享服务器实例的start_server()方法，该方法用于启动服务器
    await server_class.share_server(server_name).start_server()
