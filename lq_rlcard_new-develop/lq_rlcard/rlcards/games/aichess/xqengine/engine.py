# -*- coding: utf-8 -*-

"""
todo: 象棋[U U C I]引擎
"""

import os
import re
import sys
import time
import copy
import queue
import pathlib
import logging
import threading
import traceback
import subprocess
from attrdict import attrdict


def get_dir_path():
	return pathlib.Path(__file__).parent

logging.basicConfig(
	stream=sys.stdout,
	level=logging.DEBUG,
	format='[%(asctime)s] [%(module)s:%(lineno)d] %(levelname)s %(message)s',)
logger = logging.getLogger()


class Queue(queue.Queue):
	"""
	todo: 计算动作预测动作资源

	1.首先使用engine线程读取[u u c i]输出的行直接入队
	2.其次需要一个解析线程，单独处理[u u c i]引擎的输出，在人不干预的情况下一直读队列的输出
	3.交互
		3.1交互线程，目前为主线程，首先向队列put下面的mark，然后进入wait
		3.2当解析线程读到mark的时候，notify通知交互线程读取队列，自己进入wait等待交互完成
		3.3交互完成之后，notify通知解析线程继续解析
	"""
	mark = 'a3c63f6d-a880-4bde-a176-82af83bcf793'

	def __init__(self, maxsize=0):
		"""
		初始化参数
		"""
		super.__init__(maxsize)
		self.condition = threading.Condition()

	def __enter__(self):
		"""
		todo: 入对
		"""
		self.condition.acquire()
		self.put(self.mark)
		self.condition.wait()

	def get(self, block=True, timeout=None):
		"""
		todo: 读取
		"""
		while True:
			line = super(Queue, self).get(block=block, timeout=timeout)
			if line != self.mark:
				return line
			with self.condition:
				self.condition.notify()
				self.condition.wait()
				continue

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""
		todo: 退出
		"""
		self.condition.notify()
		self.condition.release()


class Engine(threading.Thread):
	"""
	todo: 象棋引擎
	"""
	ENGINE_BOOT = 0
	ENGINE_IDLE = 1
	ENGINE_BUSY = 2
	ENGINE_MOVE = 3
	ENGINE_INFO = 4
	NAME = '基础引擎'

	def __init__(self):
		super(Engine, self).__init__(daemon=True)
		self.index = 0
		self.running = False
		self.milli_sec = 1
		self.ues_milli_sec = True
		self.name = 'EngineThread'
		self.state = self.ENGINE_BOOT  # 启动

	def callback(self, type, data):
		pass

	def setup(self):
		pass

	def close(self):
		pass

	def position(self, fen=None):
		pass

	def go(
			self,
			depth=None,
			nodes=None,
			time=None,
			moves_to_go=None,
			increment=None,
			opp_time=None,
			opp_moves_to_go=None,
			opp_increment=None,
			draw=None,
			ponder=None):
		"""
		todo: 搜索
		"""
		return


class PipeEngine(Engine):
	"""
	引擎计算管道
	"""
	def __init__(self, file_name: Path):
		super(PipeEngine, self).__init__()
		self.pipe = None
		self.file_name = file_name
		self.dir_name = os.path.dirname(file_name)
		self.parser_thread = threading.Thread(
			target=self.parser,
			daemon=True,
			name='ParserThread'
		)
		# 引擎输出队列
		self.outlines = Queue()
		self.setup()

	def close(self):
		self.running = True
		if self.pipe:
			self.pipe.terminate()
		self.parser_thread.join()
		self.join()

	def run(self):
		"""
		引擎存有两个线程，自己读取引擎的输出，按行put到队列
		解析线程从队列读取输出，进行解析
		"""
		self.running = True
		self.parser_thread.start()
		while self.running:
			line = self.readline()
			self.outlines.put(line)

	def parse_line(self, line: str):
		"""
		解析计算结果
		"""
		pass

	def parser(self):
		"""
		解析线程
		"""
		self.running = True
		while self.running:
			line = self.outlines.get()
			try:
				self.parse_line(line)
			except Exception:
				logger.error(traceback.format_exc())

	def send_cmd(self, command):
		"""
		向计算引擎写入指令
		"""
		try:
			logger.info('command: %s', command)
			line = f'{command}\n'.encode('gbk')
			self.stdin.write(line)
			self.stdin.flush()
		except IOError as e:
			logger.error('send command error %s: ', e)
			logger.error(traceback.format_exc())

	def decode(self, line):
		"""
		解码
		"""
		try:
			return line.decode('gbk')
		except UnicodeDecodeError:
			return line.decode('utf8')

	def readline(self):
		"""
		从引擎输出中读取一行
		"""
		while 1:
			try:
				line = self.stdout.readline().strip()
				line = self.decode(line)
				if not line:
					return ''
				logger.info('output: %s', line)
				return line
			except Exception as e:
				logger.error('readline io error %s: ', e)
				logger.error(traceback.format_exc())
				continue

	def clear(self):
		"""
		清空引擎输出
		"""
		with self.outlines:
			while 1:
				try:
					self.outlines.get_nowait()
				except queue.Empty:
					return

	def setup(self):
		"""
		初始化引擎
		"""
		logger.info('open pipe %s', self.file_name)
		startupinfo = None

		if os.name == 'nt':
			startupinfo = subprocess.STARTUPINFO()
			startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

		self.pipe = subprocess.Popen(
			[str(self.filename)],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			cwd=str(self.dirname),
			startupinfo=startupinfo
		)

		self.stdin = self.pipe.stdin
		self.stdout = self.pipe.stdout
		self.stderr = self.pipe.stderr


class UUCIEngine(PipeEngine):
	"""
	todo: 中国象棋通用引擎协议(Universal Chinese Chess Protocol，简称U C C I)
	"""

	def __init__(self, file_name, callback=None):
		self.ids = attrdict()
		self.options = attrdict()
		self.callback = callback
		super().__init__(file_name)

	def setup(self):
		super().setup()
		self.send_cmd('ucci')
		while self.running:
			self.send_cmd('ucci')
			line = self.readline()
			if not line:
				continue
			try:
				self.parse_line(line)
			except Exception:
				logger.error(traceback.format_exc())

	def close(self):
		self.send_cmd('quit')
		return super().close()

	def parse_option(self, message: str):
		message = re.sub(' +', '', message)
		items = message.split()
		option = attrdict()
		key = None
		for var in items:
			if var in {'type', 'min', 'max', 'var', 'default'}:
				key = var
				continue
			elif not key:
				continue
			elif key == 'var':
				option.setdefault('vars', [])
				option.vars.append(var)
				continue
			else:
				option[key] = var
			if key == 'default':
				option.value = var
			key = None
		return option

	def set_option(self, option, value):
		if option not in self.options:
			logger.warning("option %s not supported!!!", option)
			return
		self.options[option].value = value
		command = f'setoption option: {option}, value: {value}'
		self.send_cmd(command)

	def setup_option(self, name, option):
		if name == 'use_milli_sec':
			if option.value == 'true':
				self.ues_milli_sec = True
				self.milli_sec = 1
			else:
				self.ues_milli_sec = False
				self.milli_sec = 1000

	def position(self, fen=None):
		if not fen:
			fen = ''
		mark = 'fen'
		if fen.startswith('startpos'):
			mark = ''
		command = f'position{mark}{fen}'
		self.send_cmd(command)

	def ban_moves(self, moves: list):
		if not moves:
			return
		command = f'ban moves{"".join(moves)}'
		self.send_cmd(command)

	def go(
			self,
			depth=None,
			nodes=None,
			time=None,
			moves_to_go=None,
			increment=None,
			opp_time=None,
			opp_moves_to_go=None,
			opp_increment=None,
			draw=None,
			ponder=None):
		"""
		todo: 搜索
		"""
		command = "go"
		if draw:
			command += "draw"
		elif ponder:
			command += "ponder"

		if depth:
			command += f' depth {depth}'
		elif nodes:
			command += f' nodes {depth}'
		elif time:
			time //= self.milli_sec
			command += f' time {time}'
			if increment:
				increment //= self.milli_sec
				command += f' increment {increment}'
			elif moves_to_go:
				command += f' moves_to_go {moves_to_go}'
			elif opp_time:
				opp_time //= self.millisec
				command += f' opp_time {opp_time}'
			if opp_increment:
				opp_increment //= self.milli_sec
				command += f' opp_increment {opp_increment}'
			elif opp_moves_to_go:
				command += f' opp_moves_to_go {opp_moves_to_go}'
		else:
			return

		self.send_cmd(command)

	def ponder_hit(self, draw=False):
		var=''
		if draw:
			var = 'draw'
		command = f'ponder hint {var}'
		self.send_cmd(command)

	def probe(self, fen, moves=None):
		if not moves:
			moves = []
		command = f'probe {fen} moves {" ".join(moves)}'
		self.send_command(command)

	def pop_hash(self, line: str):
		pass

	def stop(self):
		self.send_cmd('stop')

	def is_ready(self):
		self.clear()
		self.send_cmd('is_ready')
		with self.outlines:
			line = self.outlines.get()
			if line == 'ready_ok':
				return True
			return False

	def parse_line(self, line: str):
		items = line.split(maxsplit=1)
		if not items:
			return
		instruct = items[0]
		if instruct == 'bye':
			self.running = False
		if instruct in ['ready_ok', 'bye']:
			return
		if instruct == 'ucciok':
			return
		if instruct in {'id', 'option', 'info', 'pop_hash', 'best_move'}:
			tup = items[1].split(maxsplit=1)
			if instruct == 'id':
				self.ids[tup[0]] = tup[1]
				return
			if instruct == 'option':
				self.options[tup[0]] = self.parse_option(tup[1])
				self.setup_option(tup[0], self.options[tup[0]])
				return
			if instruct == 'info':
				if callable(self.callback):
					self.callback(line)
				return
			if instruct == 'pop_hash':
				if callable(self.callback):
					self.callback(line)
				return
		data = None
		if instruct == 'best_move':
			moves = items[1].split()
		elif instruct == 'no_best_move':
			pass
		else:
			logger.warning(instruct)
			return
		if callable(self.callback):
			self.callback()

def main():
	filename = dirpath / 'engines/binghe/binghe.exe'
	engine = UCCIEngine(filename)
	engine.start()