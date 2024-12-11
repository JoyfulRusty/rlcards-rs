# -*- coding: utf-8 -*-

import gym
import random
import argparse
import numpy as np
import tensorflow as tf

from collections import deque

# tf.keras.backend集成了很多常用的数学方法
tf.keras.backend.set_floatx('float64')

# 设置训练参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()

# 创建tensorboard图表
log_dir = './tensorboard/logs/'
summary_writer = tf.summary.create_file_writer(log_dir)


class ReplayBuffer:
	"""
	经验缓存池
	"""
	def __init__(self, capacity=10000):
		"""
		初始化经验缓存池大小
		"""
		self.buffer = deque(maxlen=capacity)

	def put(self, state, action, reward, next_state, done):
		"""
		向经验缓存池中添加数据
		"""
		self.buffer.append([state, action, reward, next_state, done])

	def sample(self):
		"""
		采样
		"""
		sample = random.sample(self.buffer, args.batch_size)
		states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))

		# 变换维度
		states = np.array(states).reshape(args.batch_size, -1)
		next_states = np.array(next_states).reshape(args.batch_size, -1)

		return states, actions, rewards, next_states, done

	def size(self):
		"""
		缓存池数据大小
		"""
		return len(self.buffer)


class ActionStateModel:
	"""
	动作状态模型
	"""
	def __init__(self, state_dim, action_dim):
		"""
		初始化参数
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.epsilon = args.eps

		# 构建模型
		self.model = self.create_model()

	def create_model(self):
		"""
		构建神经网络层
		"""
		state_input = tf.keras.layers.Input((self.state_dim,))
		backbone_1 = tf.keras.layers.Dense(32, activation='relu')(state_input)
		backbone_2 = tf.keras.layers.Dense(16, activation='relu')(backbone_1)
		value_output = tf.keras.layers.Dense(1)(backbone_2)
		advantage_output = tf.keras.layers.Dense(self.action_dim)(backbone_2)
		output = tf.keras.layers.Add()([value_output, advantage_output])
		model = tf.keras.Model(state_input, output)
		model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(args.lr))

		return model

	def predict(self, state):
		"""
		预测
		"""
		return self.model.predict(state)

	def get_action(self, state):
		"""
		动作获取
		"""
		state = np.reshape(state, [1, self.state_dim])
		self.epsilon *= args.eps_decay
		self.epsilon = max(self.epsilon, args.eps_min)

		# Q值
		q_value = self.predict(state)[0]
		if np.random.random() < self.epsilon:
			return random.randint(0, self.action_dim - 1)
		return np.argmax(q_value)

	def train(self, states, targets):
		"""
		训练
		"""
		self.model.fit(states, targets, epochs=1, verbose=0)


class Agent:
	"""
	游戏环境代理
	"""
	def __init__(self, env):
		"""
		初始化游戏环境参数
		"""
		self.env = env
		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.n
		# 构建模型
		self.model = ActionStateModel(self.state_dim, self.action_dim)
		# 目标模型用于保存训练权重
		self.target_model = ActionStateModel(self.state_dim, self.action_dim)
		# 更新权重
		self.target_update()
		self.buffer = ReplayBuffer()

	def target_update(self):
		"""
		更新目标模型权重
		"""
		weights = self.model.model.get_weights()
		self.target_model.model.set_weights(weights)

	def replay(self):
		"""
		经验缓存池，预测下一个Q值
		"""
		for _ in range(10):
			states, actions, rewards, next_states, done = self.buffer.sample()
			targets = self.target_model.predict(states)

			# 下一个Q值
			next_q_values = self.target_model.predict(next_states).max(axis=1)
			targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
			self.model.train(states, targets)

	def train(self, max_episodes=1000):
		"""
		训练
		"""
		for epoch in range(max_episodes):
			done, total_reward = False, 0
			state = self.env.reset()[0]
			while not done:
				action = self.model.get_action(state)
				next_state, reward, done, _, _ = self.env.step(action)
				# 向缓存池添加数据
				self.buffer.put(state, action, reward * 0.01, next_state, done)
				# 计算总奖励和下一个状态
				total_reward += reward
				state = next_state

			# 经验缓存池
			if self.buffer.size() >= args.batch_size:
				self.replay()

			# 更新目标值
			self.target_update()
			print('EP{} EpisodeReward={}'.format(epoch, total_reward))

def main():
	"""
	启动
	"""
	env_name = 'CartPole-v1'
	# 创建环境
	env = gym.make(env_name, render_mode="human")
	agent = Agent(env)
	agent.train(max_episodes=1000)

if __name__ == "__main__":
	main()
