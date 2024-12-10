# -*- coding: utf-8 -*-

import os
import sys

import atexit
import signal

def daemonize(pidfile, *, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null', on_exit=None):
	"""
	将当前进程守护化，并写入PID文件。
	:param pidfile: PID文件路径
	:param stdin: 守护进程的标准输入重定向路径，默认为/dev/null
	:param stdout: 守护进程的标准输出重定向路径，默认为/dev/null
	:param stderr: 守护进程的错误输出重定向路径，默认为/dev/null
	:param on_exit: 守护进程退出时执行的回调函数，默认为None
	"""
	if os.path.exists(pidfile):
		raise RuntimeError("Already running")
	# First fork (detaches from parent)
	# 第一次fork，生成子进程，然后退出主进程
	# 从父进程fork一个子进程出来
	try:
		# 这是第二次fork，用于创建另一个子进程。与第一次fork不同，这次fork会返回一个负值，表示失败。如果fork成功，则返回子进程的ID
		# 在Unix和类Unix系统（如Linux）中，fork()函数用于创建新的进程。fork()函数会返回两次，一次在父进程（返回子进程的ID），一次在子进程（返回0）。
		# fork() > 0代表的是父进程。当fork()在父进程中调用时，它会返回一个新的进程ID（一个正整数），这个新的进程ID是唯一的，用于标识新创建的子进程。
		# 父进程可以通过这个进程ID来控制子进程，例如等待子进程结束、获取子进程的返回值等
		if os.fork() > 0:
			# 如果fork失败，则抛出一个SystemExit异常，退出主进程。这样，主进程就会立即退出，而子进程将继续运行
			raise SystemExit(0)  # 退出主进程  Parent exit
	except OSError as e:
		# 如果fork失败，则捕获OSError异常，并打印错误信息
		print(f"OSError#1-{e}")
		# 如果fork失败，则抛出一个RuntimeError异常，表示fork失败
		raise RuntimeError('fork #1 failed.')

	# 子进程默认继承父进程的工作目录，最好是变更到根目录，否则会影响文件系统的卸载
	# os.chdir('/')
	# 子进程默认继承父进程的umask(文件权限掩码)，重设为0(完全控制)，以避免程序读写文件
	os.umask(0)
	# 让子进程成为新的会话组长和进程组长
	os.setsid()

	# Second fork (relinquish session leadership)
	# 此时，孙子进程已经是守护进程了，接下来重定向标准输入、输出、错误的描述符(是重定向而不是关闭, 这样可以避免程序在 print 的时候出错)
	try:
		# 用于创建另一个子进程。与第一次fork不同，这次fork会返回一个负值，表示失败。如果fork成功，则返回子进程的ID
		if os.fork() > 0:
			# 如果fork失败，则抛出一个SystemExit异常，退出主进程。这样，主进程就会立即退出，而子进程将继续运行
			raise SystemExit(0)  # 退出主进程  Parent exit
	except OSError as e:
		# 如果fork失败，则捕获OSError异常，并打印错误信息
		print(f"OSError#2-{e}")
		# 如果fork失败，则抛出一个RuntimeError异常，表示fork失败
		raise RuntimeError('fork #2 failed.')

	# 先刷新缓冲区
	# Flush I/O buffers
	sys.stdout.flush()
	sys.stderr.flush()

	# Replace file descriptors for stdin, stdout, and stderr
	# dup2函数原子化地关闭和复制文件描述符，重定向到/dev/nul，即丢弃所有输入输出
	with open(stdin, 'rb', 0) as f:
		os.dup2(f.fileno(), sys.stdin.fileno())
	with open(stdout, 'ab', 0) as f:
		os.dup2(f.fileno(), sys.stdout.fileno())
	with open(stderr, 'ab', 0) as f:
		os.dup2(f.fileno(), sys.stderr.fileno())

	# Arrange to have the PID file removed on exit/signal
	# 注册退出函数，进程异常退出时移除pid文件
	atexit.register(lambda: os.remove(pidfile))

	# Signal handler for termination (required)
	# 用于终止的信号处理程序(必需)
	def sigterm_handler(*args):
		raise SystemExit(1)

	signal.signal(signal.SIGTERM, on_exit if on_exit else sigterm_handler)