# -*- coding: utf-8 -*-

import gym
import argparse
import numpy as np

import tensorflow as tf

# tf.keras.backend集成了很多常用的数学方法
tf.keras.backend.set_floatx('float64')

# 设置训练参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()

# 创建tensorboard图表
log_dir = './tensorboard/logs/'
summary_writer = tf.summary.create_file_writer(log_dir)


class Actor:
	"""
	Actor训练代理
	"""
	def __init__(self, state_dim, action_dim, action_bound, std_bound):
		"""
		初始化训练神经网络参数
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.std_bound = std_bound
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.actor_lr)

	def create_model(self):
		"""
		构建神经网络层
		"""
		state_input = tf.keras.layers.Input((self.state_dim,))
		dense_1 = tf.keras.layers.Dense(32, activation='relu')(state_input)
		dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)
		out_mu = tf.keras.layers.Dense(self.action_dim, activation='tanh')(dense_2)
		mu_output = tf.keras.layers.Lambda(lambda x: x * self.action_bound)(out_mu)
		std_output = tf.keras.layers.Dense(self.action_dim, activation='softplus')(dense_2)

		return tf.keras.models.Model(state_input, [mu_output, std_output])

	def get_action(self, state):
		"""
		动作获取
		"""
		state = np.reshape(state, [1, self.state_dim])
		mu, std = self.model.predict(state)
		mu, std = mu[0], std[0]

		return np.random.normal(mu, std, size=self.action_dim)

	def normal_loss(self, mu, std, action):
		"""
		归一化处理
		"""
		std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
		var = std ** 2
		loss_polity = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)

		return tf.reduce_sum(loss_polity, 1, keepdims=True)

	def compute_loss(self, mu, std, actions, advantages):
		"""
		计算损失值
		"""
		loss_polity = self.normal_loss(mu, std, actions)
		loss_polity = loss_polity * advantages

		return tf.reduce_sum(-loss_polity)

	def train(self, states, actions, advantages):
		"""
		训练
		"""
		# 根据某个函数的输入遍历来计算其导数
		# tf.GradientTape()上下文中执行的所有操作都记录在一个磁带(tape)上
		# 基于这个磁带和每次操作产生的导数，使用反向微分法来计算这些被记录的函数的导数
		with tf.GradientTape() as tape:
			mu, std = self.model(states, training=True)
			loss = self.compute_loss(mu, std, actions, advantages)

		# 获取损失值对各个变量的偏导数之和，即梯度,大小len(input_data)
		# compute_gradients()返回的值作为输入参数对variable进行更新
		grads = tape.gradient(loss, self.model.trainable_variables)

		# 进行梯度裁剪apply_gradients or minimize
		# 调用minimize的时候是干了两件事情，compute_gradients和apply_gradients，
		# 这样写更为方便，但是如果要进行梯度的裁剪，梯度的选择就要写成上面的形式
		self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

		return loss

class Critic:
	"""
	Critic评判代理
	"""
	def __init__(self, state_dim):
		"""
		初始化评判代理神经网络参数
		"""
		self.state_dim = state_dim
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.critic_lr)

	def create_model(self):
		"""
		构建神经网络模型
		"""
		critic_model = tf.keras.Sequential([
			tf.keras.layers.Input((self.state_dim, )),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(16, activation='relu'),
			tf.keras.layers.Dense(1, activation='linear')
		])

		return critic_model

	@staticmethod
	def compute_loss(v_pred, td_targets):
		"""
		计算损失值
		"""
		mse = tf.keras.losses.MeanSquaredError()

		return mse(td_targets, v_pred)

	def train(self, states, td_targets):
		"""
		训练模型
		"""
		with tf.GradientTape() as tape:
			v_pred = self.model(states, training=True)
			assert v_pred.shape == td_targets.shape

			# 计算损失值
			loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))

		grads = tape.gradient(loss, self.model.trainable_variables)
		self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

		return loss

class Agent:
	"""
	游戏环境
	"""
	def __init__(self, env):
		"""
		初始化游戏环境参数
		"""
		self.env = env
		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		self.action_bound = self.env.action_space.shape[0]
		self.std_bound = [1e-2, 1.0]

		# 创建训练网络和评价网络
		self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
		self.critic = Critic(self.state_dim)

	def td_target(self, reward, next_state, done):
		"""
		TD目标
		"""
		if done:
			return reward
		v_value = self.critic.model.predict(np.reshape(next_state, [1, self.state_dim]))

		return np.reshape(reward + args.gamma * v_value[0], [1, 1])

	@staticmethod
	def advantage(td_target, baselines):
		"""
		有利点或奖励
		"""
		return td_target - baselines

	@staticmethod
	def list_to_batch(list_data):
		"""
		遍历批次数据
		"""
		batch = list_data[0]
		for elem in list_data[1:]:
			batch = np.append(batch, elem, axis=0)
		return batch

	def train(self, max_episodes=1000):
		"""
		启动训练
		"""
		# max_episodes最大训练批次数量
		for epoch in range(max_episodes):
			state_batch = []
			action_batch = []
			td_target_batch = []
			advantage_batch = []
			episode_reward, done = 0, False

			# 状态数据
			state = self.env.reset()
			state = state[0]

			while not done:
				action = self.actor.get_action(state)
				action = np.clip(action, -self.action_bound, self.action_bound)

				# 更新状态和动作数据
				next_state, reward, done, _, _ = self.env.step(action)
				state = np.reshape(state, [1, self.state_dim])
				action = np.reshape(action, [1, self.action_dim])
				next_state = np.reshape(next_state, [1, self.state_dim])
				reward = np.reshape(reward, [1, 1])

				td_target = self.td_target((reward + 8) / 8, next_state, done)
				advantage = self.advantage(td_target, self.critic.model.predict(state))
				state_batch.append(state)
				action_batch.append(action)
				td_target_batch.append(td_target)
				advantage_batch.append(advantage)

				if len(state_batch) >= args.update_interval or done:
					states = self.list_to_batch(state_batch)
					actions = self.list_to_batch(action_batch)
					td_targets = self.list_to_batch(td_target_batch)
					advantages = self.list_to_batch(advantage_batch)

					actor_loss = self.actor.train(states, actions, advantages)
					critic_loss = self.critic.train(states, td_targets)

					with summary_writer.as_default():
						tf.summary.scalar("actor_loss: ", actor_loss, step=epoch)
						tf.summary.scalar("critic_loss: ", critic_loss, step=epoch)

					state_batch = []
					action_batch = []
					td_target_batch = []
					advantage_batch = []

				episode_reward += reward[0][0]
				state = next_state[0]

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