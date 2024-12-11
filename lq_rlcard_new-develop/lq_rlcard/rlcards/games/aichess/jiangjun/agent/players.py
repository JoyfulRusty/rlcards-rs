# -*- coding: utf-8 -*-

import random
import asyncio
import logging
import numpy as np

from config import conf
from agent import mcts_async
from asyncio.queues import Queue
from collections import namedtuple
from env.cchess_env import CchessEnv
from env.cchess_env_c import CchessEnvC
from lib.game_convert import board_arr_2_net_input
from env.cchess_env import create_uci_labels, BaseChessBoard

labels = create_uci_labels()
uci_labels = create_uci_labels()
QueueItem = namedtuple('QueueItem', "feature future")


def flipped_uci_labels(param):
	"""
	反转uci标签
	"""
	def repl(x):
		return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])
	return [repl(x) for x in param]


class Player:
	"""
	玩家
	"""
	def __init__(self, side):
		"""
		初始化玩家参数
		"""
		assert(side in ['w', 'b'])
		self.side = side

	def make_move(self, state):
		"""
		制作移动
		"""
		assert(state.currentplayer == self.side)
		pass

	def oppo_make_move(self, single_move):
		"""
		对立移动
		"""
		pass


class NetWorkPlayer:
	"""
	神经网络玩家
	"""
	def __init__(
			self,
			side,
			network,
			debugging=True,
			n_play_out=800,
			search_threads=1,
			virtual_loss=0.02,
			policy_loop_arg=True,
			c_p_uct=5,
			d_noise=False,
			temp_round=conf.SelfPlayConfig.train_temp_round,
			can_surrender=False,
			surrender_threshold=-0.99,
			allow_legacy=False,
			repeat_noise=True,
			is_self_play=False,
			play=False,
			ma_service=False):
		"""
		初始化属性参数
		"""
		super(NetWorkPlayer, self).__init__(side)
		self.network = network
		self.debugging = debugging
		loop = None
		if ma_service:
			new_loop = asyncio.new_event_loop()
			asyncio.set_event_loop(new_loop)
			loop = asyncio.get_event_loop()
		self.queue = Queue(400, loop=loop)
		self.temp_round = temp_round
		self.can_surrender = can_surrender
		self.allow_legacy = allow_legacy
		self.surrender_threshold = surrender_threshold
		self.repeat_noise = repeat_noise
		self.mcts_policy = mcts_async.MCTS(
			self.policy_value_fn_queue,
			n_play_out=n_play_yout,
			search_threads=search_threads,
			virtual_loss=virtual_loss,
			policy_loop_arg=policy_loop_arg,
			c_p_uct=c_p_uct,
			d_noise=d_noise,
			play=play,
		)
		self.is_self_play = is_self_play

	async def push_queue(self, features, loop):
		"""
		添加到队列
		"""
		future = loop.create_future()
		item = QueueItem(features, future)
		await self.queue.put(item)
		return future

	async def prediction_worker(self, mcts_policy_async):
		"""
		预测worker
		"""
		(sess, graph), ((X, training), (net_softmax, value_head)) = self.network
		q = self.queue
		while mcts_policy_async.num_preceed < mcts_policy_async._n_play_out:
			if q.empty():
				await asyncio.sleep(1e-3)
				continue
			item_list = [q.get_nowait() for _ in range(q.qsize())]
			features = np.concatenate([item.feature for item in item_list], axis=0)
			action_prob, value = sess.run([net_softmax, value_head], feed_dict={X: features, training: False})
			for p, v, item in zip(action_prob, value, item_list):
				item.future.set_result((p, v))

	async def policy_value_fn_queue(self, state, loop):
		"""
		计算队列中的策略价值
		"""
		bb = BaseChessBoard(state.state_str)
		state_str = bb.get_board_arr()
		net_x = np.transpose(board_arr_2_net_input(state_str, state.get_curr_player()), [1, 2, 0])
		net_x = np.expand_dims(net_x, 0)

		future = await self.push_queue(net_x, loop)
		await future

		policy_out, val_out = future.result()
		policy_out, val_out = policy_out, val_out[0]
		if conf.SelfPlayConfig.py_env:
			legal_move = CchessEnv.get_legal_moves(state.state_str, state.get_curr_player())
		else:
			legal_move = CchessEnv.get_legal_action(state.state_str, state.get_curr_player())
		legal_move = set(legal_move)
		legal_move_b = set(flipped_uci_labels(legal_move))
		action_prob = []
		if state.currentplayer == 'b':
			for single_move, prob in zip(uci_labels, policy_out):
				if single_move in legal_move_b:
					single_move = flipped_uci_labels([single_move])[0]
					action_prob.append((single_move, prob))
		else:
			for single_move, prob in zip(uci_labels, policy_out):
				if single_move in legal_move:
					action_prob.append((single_move, prob))
		return action_prob, val_out

	def get_random_policy(self, policies):
		"""
		获取随机策略
		"""
		sum_num = sum([i[1] for i in policies])
		rand_num = random.random() * sum_num
		tmp = 0
		for val, pos in policies:
			tmp += pos
			if tmp > rand_num:
				return val

	def make_move(self, state, actual_move=True, infer_mode=False, allow_legacy=False, no_act=None):
		"""
		移动
		"""
		assert(state.currentplayer == self.side)
		if state.move_number < self.temp_round or (self.repeat_noise and state.maxrepeat > 1):
			temp = 1
		else:
			temp = 1e-4

		if state.move_number >= self.temp_round and self.is_self_play is True:
			can_apply_d_noise = True
		else:
			can_apply_d_noise = False
		info = []
		if infer_mode:
			acts, act_prob, info = self.mcts_policy.get_move_prob(
				state,
				temp=temp,
				verbose=False,
				predict_workers=[self.prediction_worker(self.mcts_policy)],
				can_apply_d_noise=can_apply_d_noise,
				infer_mode=infer_mode,
				no_act=no_act
			)
		else:
			acts, act_prob = self.mcts_policy.get_move_prob(
				state,
				temp=temp,
				verbose=False,
				predict_workers=[self.prediction_worker(self.mcts_policy)],
				can_apply_d_noise=can_apply_d_noise,
				no_act=no_act
			)
		# 冻结参数吗？
		if not acts:
			if infer_mode:
				return None, None, None
			return None, None
		policies, score = list(zip(acts, act_prob)), self.mcts_policy._root._Q
		score = -score
		# 1表示赢，-1表示输
		if score < self.surrender_threshold and self.can_surrender:
			return None, score
		# 获取随机策略
		single_move = self.get_random_policy(policies)
		if actual_move:
			state.do_move(single_move)
			self.mcts_policy.update_with_move(single_move, allow_legacy=allow_legacy)
		if infer_mode:
			return single_move, score, info
		return single_move, score

	def oppo_make_move(self, single_move, allow_legacy=False):
		"""
		对立移动
		"""
		self.mcts_policy.update_with_move(single_move, allow_legacy=allow_legacy)