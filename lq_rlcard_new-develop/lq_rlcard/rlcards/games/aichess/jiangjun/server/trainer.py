# -*- coding: utf-8 -*-

import os
import time
import logging
import numpy as np
import tensorflow as tf

from agent.resnet import get_model
from server.sampler import Sampler
from server.data_loader import ElePreloader
from env.cchess_env import create_uci_labels
from config.conf import TrainingConfig, ResourceConfig
from server.utils import get_latest_weight_path, cycle_lr


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

class Trainer:
	"""
	训练
	"""
	def __init__(self):
		"""
		初始化参数
		"""
		self.GPU_CORE = [0, 1, 2, 3]
		self.labels = create_uci_labels()
		# 创建模型文件夹
		if not os.path.exists(ResourceConfig.model_dir):
			os.makedirs(ResourceConfig.model_dir)
		if ResourceConfig.restore_path is not None:
			self.restore_model = ResourceConfig.restore_path
		else:
			# 加载最后权重路径
			self.restore_model = get_latest_weight_path()
		# 采样
		self.sampler = Sampler()

	def train(self):
		"""
		训练
		"""
		while True:
			try:
				(sess, graph), (
					(X, training),
					(
						net_softmax,
						value_head,
						train_op_multi_arg,
						(train_op_policy, train_op_value),
						policy_loss, accuracy_select,
						global_step,
						value_loss,
						next_move,
						learning_rate,
						score,
						multi_get_loss,
						merged
					)
				) = get_model(
					self.restore_model,
					self.labels,
					gpu_core=self.GPU_CORE,
					filters=TrainingConfig.network_filters,
					num_res_layers=TrainingConfig.network_layers,
					extrav2=True,
					batch_size=TrainingConfig.batch_size
				)

				writer = tf.summary.FileWriter(ResourceConfig.tensorboard_dir, sess.graph)
				model_restored = self.restore_model.split('/')[-1].split('.')[0]
				step = int(model_restored.split('_')[-1])
				logging.info('restore model {}, global step {}'.format(model_restored, step))
				logging.info('start training')
				while True:
					t = time.time()
					file_list = self.sampler.sample()
					train_set = ElePreloader(data_list=file_list, batch_size=TrainingConfig.batch_size)
					learning_rate_this_step = cycle_lr(step)
					batch_x, batch_y, batch_v, one_finish_sum = trainset.load(ResourceConfig.num_process)
					batch_v = np.expand_dims(np.nan_to_num(batch_v), 1)
					if step % 10 == 0:
						logging.info('load data {}'.format(time.time() - t))
					with graph.as_default():
						summary, _, step_value_loss, step_val_predict, step_policy_loss, step_acc_move, step_value, step_total_loss = \
							sess.run(
								[merged, train_op_multitarg, value_loss, value_head, policy_loss, accuracy_select,
								 global_step, multitarget_loss],
								feed_dict={
									X: batch_x,
									learning_rate: learning_rate_this_step,
									training: True,
									score: batch_v,
									nextmove: batch_y,
								}
							)
					step += 1
					writer.add_summary(summary, global_step=step)
					if step % 10 == 0:
						logging.info('train data {}'.format(time.time() - t))
						logging.info('training step {}'.format(step))
					stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
					if step % TrainingConfig.saver_step == 0:
						with graph.as_default():
							saver = tf.train.Saver(var_list=tf.global_variables())
							saver.save(sess, "{}/{}_{}".format(ResourceConfig.model_dir, stamp, step))
							logging.info('----------- save weights {}'.format(stamp))
			except BaseException:
				logging.info('train error one time oh my god')