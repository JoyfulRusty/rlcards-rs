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
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.05)
parser.add_argument('--train_start', type=int, default=2000)

args = parser.parse_args()

# 创建tensorboard图表
log_dir = './tensorboard/logs/'
summary_writer = tf.summary.create_file_writer(log_dir)

class ReplayBuffer:
	"""
	经验回放池
	"""
	def __init__(self, capacity=20000):
		"""
		初始化缓存队列buffer
		"""
		self.buffer = deque(maxlen=capacity)

	def put(self, state, action, reward, next_state, done):
		"""
		在经验回放池添加数据
		"""
		self.buffer.append([state, action, reward, next_state, done])

	def sample(self):
		"""
		采样批次数据大小
		"""
		sample = random.sample(self.buffer, args.batch_size)
		states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
		states = np.array(states).reshape(args.batch_size, -1)
		next_states = np.array(next_states).reshape(args.batch_size, -1)

		return states, actions, rewards, next_states, done

	def size(self):
		"""
		缓存大小
		"""
		return len(self.buffer)


class Actor:
	"""
	Actor行动训练代理
	"""
	def __init__(self, state_dim, action_dim, action_bound):
		"""
		初始化参数
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.actor_lr)

	def create_model(self):
		"""
		构建神经网络层
		"""
		actor_model = tf.keras.Sequential([
			tf.keras.layers.Input((self.state_dim,)),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(self.action_dim, activation='tanh'),
			tf.keras.layers.Lambda(lambda x: x * self.action_bound)
		])

		return actor_model

	def train(self, states, q_grads):
		"""
		训练
		"""
		with tf.GradientTape() as tape:
			grads = tape.gradient(self.model(states), self.model.trainable_variables, -q_grads)
		self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

	def predict(self, state):
		"""
		预测
		"""
		return self.model.predict(state)

	def get_action(self, state):
		"""
		获取动作
		"""
		state = np.reshape(state, [1, self.state_dim])
		return self.model.predict(state)[0]


class Critic:
	"""
	Critic评判代理
	"""
	def __init__(self, state_dim, action_dim):
		"""
		初始化评判代理神经网络参数
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.critic_lr)

	def create_model(self):
		"""
		构建神经网络模型
		"""
		state_input = tf.keras.layers.Input((self.state_dim, ))
		s1 = tf.keras.layers.Dense(64, activation='relu')(state_input)
		s2 = tf.keras.layers.Dense(32, activation='relu')(s1)
		action_input = tf.keras.layers.Input((self.action_dim,))
		a1 = tf.keras.layers.Dense(32, activation='relu')(action_input)
		c1 = tf.keras.layers.concatenate([s2, a1], axis=-1)
		c2 = tf.keras.layers.Dense(16, activation='relu')(c1)
		output = tf.keras.layers.Dense(1, activation='linear')(c2)

		return tf.keras.Model([state_input, action_input], output)

	def predict(self, inputs):
		"""
		预测
		"""
		return self.model.predict(inputs)

	def q_grads(self, states, actions):
		"""
		求导梯度
		"""
		actions = tf.convert_to_tensor(actions)
		with tf.GradientTape() as tape:
			# 显式指定需要求导的变量
			tape.watch(actions)
			q_values = self.model([states, actions])
			q_values = tf.squeeze(q_values)

		return tape.gradient(q_values, actions)

	@staticmethod
	def compute_loss(v_pred, td_targets):
		"""
		计算损失值
		"""
		mse = tf.keras.losses.MeanSquaredError()

		return mse(td_targets, v_pred)

	def train(self, states, actions, td_targets):
		"""
		训练
		"""
		with tf.GradientTape() as tape:
			v_pred = self.model([states, actions], training=True)
			assert v_pred.shape == td_targets.shape
			loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))

		grads = tape.gradient(loss, self.model.trainable_variables)
		self.opt.apply_gradoents(zip(grads, self.model.trainable_variables))

		return loss

class Agent:
	"""
	游戏环境代理
	"""
	def __init__(self, env):
		"""
		初始化游戏环境代理参数
		"""
		self.env = env
		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		self.action_bound = self.env.action_space.high[0]

		# 创建缓存池buffer
		self.buffer = ReplayBuffer()

		# 构建模型
		self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
		self.critic = Critic(self.state_dim, self.action_dim)

		# 目标actor and critic
		self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound)
		self.target_critic = Critic(self.state_dim, self.action_dim)

		# 权重
		actor_weights = self.actor.model.get_weights()
		critic_weight = self.critic.model.get_weights()

		# 设置权重
		self.target_actor.model.set_weights(actor_weights)
		self.target_critic.model.set_weights(critic_weight)

	def target_update(self):
		"""
		更新目标
		"""
		actor_weights = self.actor.model.get_weights()
		t_actor_weights = self.target_actor.model.get_weights()
		critic_weights = self.critic.model.get_weights()
		t_critic_weights = self.target_critic.model.get_weights()

		for i in range(len(actor_weights)):
			t_actor_weights[i] = args.tau * actor_weights[i] + (1 - args.tau) * t_actor_weights[i]

		for i in range(len(critic_weights)):
			t_critic_weights[i] = args.tau * critic_weights[i] + (1 - args.tau) * t_critic_weights[i]

		self.target_actor.model.set_weights(t_actor_weights)
		self.target_critic.model.set_weights(t_critic_weights)

	@staticmethod
	def td_target(rewards, q_values, done):
		"""
		td时序差分目标
		"""
		targets = np.asarray(q_values)
		for i in range(q_values.shape[0]):
			if done[i]:
				targets[i] = rewards[i]
			else:
				targets[i] = args.gamma * q_values[i]
		return targets

	@staticmethod
	def list_to_batch(list_data):
		"""
		遍历批次数据
		"""
		batch = list_data[0]
		for elem in list_data[1:]:
			batch = np.append(batch, elem, axis=0)
		return batch

	@staticmethod
	def ou_noise(x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
		"""
		噪音值
		"""
		return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

	def replay(self):
		"""
		重放
		"""
		for _ in range(10):
			states, actions, rewards, next_states, done = self.buffer.sample()
			target_q_values = self.target_critic.predict([next_states, self.target_actor.predict(next_states)])
			td_targets = self.td_target(rewards, target_q_values, done)

			# 训练
			loss = self.critic.train(states, actions, td_targets)

			with summary_writer.as_default():
				tf.summary.scalar("critic_loss: ", loss, step=_)

			actions = self.actor.predict(states)
			grads = self.critic.q_grads(states, actions)
			grads = np.array(grads).reshape((-1, self.action_dim))
			self.actor.train(states, grads)
			self.target_update()

	def train(self, max_episodes=1000):
		"""
		训练
		"""
		for epoch in range(max_episodes):
			episode_reward, done = 0, False
			state = self.env.reset()[0]
			bg_noise = np.zeros(self.action_dim)

			# 更新动作和训练
			while not done:
				# 动作选取
				action = self.actor.get_action(state)
				noise = self.ou_noise(bg_noise, dim=self.action_dim)
				action = np.clip(action + noise, -self.action_bound, self.action_bound)

				# 更新状态和动作数据
				next_state, reward, done, _, _ = self.env.step(action)
				self.buffer.put(state, action, (reward + 8) / 8, next_state, done)

				bg_noise = noise
				episode_reward += reward
				state = next_state

			if self.buffer.size() > args.batch_size and self.buffer.size() >= args.train_start:
				self.replay()

			print('EP{} EpisodeReward={}'.format(epoch, episode_reward))


def main():
	"""
	启动
	"""
	env_name = 'Pendulum-v1'
	# 创建环境
	env = gym.make(env_name, render_mode="human")
	agent = Agent(env)
	agent.train()

if __name__ == "__main__":
	main()