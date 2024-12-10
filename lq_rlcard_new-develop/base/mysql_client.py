# -*- coding: utf-8 -*-

import asyncio
import aiomysql

from aiomysql import Pool
from aiomysql.cursors import DictCursor

class MysqlClient:
	"""
	用于执行 SQL 查询和处理数据库连接的类
	"""
	CONN_MAP = {}

	def __init__(self, db_conf, logs=None):
		"""
		初始化sql-client配置参数
		"""
		self.__ping = 10
		self.__logs = logs
		self.__conn = None
		self.__db_conf = db_conf

	@classmethod
	def init(cls, db_conf, logs=None):
		"""
		初始化数据库配置
		"""
		if not all(key in db_conf for key in ['host', 'port', 'db', 'user', 'password']):
			raise Exception('db_conf must contain host, port, db, user, password')
		conf_name = f"{db_conf['host']}-{db_conf['port']}-{db_conf['db']}"
		clt = cls.CONN_MAP.get(conf_name)
		if not clt:
			clt = cls(db_conf, logs=logs)
			cls.CONN_MAP[conf_name] = clt
		return clt

	async def init_msql_pool(self):
		"""
		初始化数据库连接池
		"""
		try:
			if not self.__conn:
				self.__db_conf.update({"cursorclass": DictCursor})
				self.__conn: Pool = await aiomysql.create_pool(**self.__db_conf)
			self.__conn and await self.__logs.info_log(f"mysql init connection pool success: {self.__conn}")
			self.__conn and await self.__logs.info_log()
		except Exception as err:
			await self.__logs.error_log(f"mysql init connection pool error: {err}")

	def check_msql_loop_ping(self, loop=None):
		"""
		检测并设置数据库连接池循环
		"""
		if not loop:
			loop = asyncio.get_event_loop()
		loop.create_task(self.__msql_loop_ping())

	async def __msql_loop_ping(self):
		"""
		ping-mysql连接池
		"""
		while 1:
			if self.__conn:
				await self.__logs.info_log(f"mysql ping connection pool success: {self.__conn}")
			else:
				try:
					if not self.__conn:
						await self.init_msql_pool()
				except Exception as err:
					await self.__logs.error_log(f"mysql ping connection pool has failed, will retry in {self.__ping} seconds: {err}")
					self.__conn = None
			await asyncio.sleep(self.__ping)

	async def close_db_pool(self):
		"""
		关闭数据库连接池
		"""
		try:
			if self.__conn:
				self.__conn.close()
				await self.__conn.wait_closed()
		except Exception as err:
			await self.__logs.error_log(f"mysql close pool error: {err}")

	async def execute_by_sql(self, sql: str):
		"""
		执行SQL语句
		"""
		async with self.__conn.acquire() as conn:
			async with conn.cursor() as cur:
				try:
					await cur.execute(sql)
					return await cur.fetchall()
				except Exception as err:
					await self.__logs.error_log(f"execute_by_sql error: {err}")

	async def query_by_one(self, sql: str):
		"""
		查询单条数据
		"""
		async with self.__conn.acquire() as conn:
			async with conn.cursor() as cur:
				try:
					await cur.execute(sql, {})
					return await cur.fetchone()
				except Exception as err:
					await self.__logs.error_log(f"query_by_one error: {err}")

	async def insert_db_data_by_sql(self, sql: str, params: list):
		"""
		插入数据到数据库
		"""
		async with self.__conn.acquire() as conn:
			async with conn.cursor() as cursor:
				await cursor.connection.begin()
				try:
					await cursor.executemany(sql, params)
					await cursor.connection.commit()
				except Exception as e:
					await self.__logs.error_log(f'insert_chat_by_sql{e}')
					await cursor.connection.rollback()
					return False