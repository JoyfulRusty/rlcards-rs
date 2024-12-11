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
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--lambda_ratio', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=3)

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
		初始化参数
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.std_bound = std_bound
		self.action_bound = action_bound

		# 构建模型、优化器
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

	def normal_loss(self, mu, std, action):
		"""
		归一化处理
		"""
		std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
		var = std ** 2
		loss_polity = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)

		return tf.reduce_sum(loss_polity, 1, keepdims=True)

	def get_action(self, state):
		"""
		动作获取
		"""
		state = np.reshape(state, [1, self.state_dim])
		mu, std = self.model.predict(state)
		action = np.random.normal(mu[0], std[0], size=self.action_dim)
		action = np.clip(action, -self.action_bound, self.action_bound)
		loss_policy = self.normal_loss(mu, std, action)

		return loss_policy, action

	@staticmethod
	def compute_loss(loss_old_polity, loss_new_polity, ges):
		"""
		计算损失值
		"""
		ratio = tf.exp(loss_new_polity - tf.stop_gradient(loss_old_polity))
		ges = tf.stop_gradient(ges)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
		surrogate = -tf.minimum(ratio * ges, clipped_ratio * ges)

		return tf.reduce_sum(surrogate)

	def train(self, loss_old_polity, states, actions, ges):
		"""
		训练
		"""
		with tf.GradientTape() as tape:
			mu, std = self.model(states, training=True)
			loss_new_polity = self.normal_loss(mu, std, actions)
			loss = self.compute_loss(loss_old_polity, loss_new_polity, ges)

		grads = tape.gradient(loss, self.model.trainable_variables)
		self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

		return loss


class Critic:
	"""
	Critic评判代理
	"""
	def __init__(self, state_dim):
		"""
		初始化参数
		"""
		self.state_dim = state_dim

		# 构建模型、优化器
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.critic_lr)

	def create_model(self):
		"""
		构建神经网络层
		"""
		critic_model = tf.keras.Sequential([
			tf.keras.layers.Input((self.state_dim,)),
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
		训练
		"""
		with tf.GradientTape() as tape:
			v_pred = self.model(states, training=True)
			assert v_pred.shape == td_targets.shape
			loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))

		grads = tape.gradient(loss, self.model.trainable_variables)
		self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

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
		self.std_bound = [1e-2, 1.0]

		self.actor_opt = tf.keras.optimizers.Adam(args.actor_lr)
		self.critic_opt = tf.keras.optimizers.Adam(args.critic_lr)

		# 构建模型
		self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
		self.critic = Critic(self.state_dim)

	def gae_target(self, rewards, v_values, next_v_value, done):
		"""
		目标值(gae_target)
		"""
		n_step_targets = np.zeros_like(rewards)
		gae = np.zeros_like(rewards)
		gae_cumulative = 0  # 累计
		forward_val = 0

		if not done:
			forward_val = next_v_value

		for k in reversed(range(0, len(rewards))):
			delta = rewards[k] + args.gamma * forward_val - v_values[k]
			gae_cumulative = args.gamma * args.lambda_ratio * gae_cumulative + delta
			gae[k] = gae_cumulative
			forward_val = v_values[k]
			n_step_targets[k] = gae[k] + v_values[k]

		return gae, n_step_targets

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
		训练
		"""
		for epoch in range(max_episodes):
			state_batch = []
			action_batch = []
			reward_batch = []
			old_policy_batch = []
			episodes_reward, done = 0, False
			state = self.env.reset()[0]

			while not done:
				log_old_policy, action = self.actor.get_action(state)

				next_state, reward, done, _, _ = self.env.step(action)

				state = np.reshape(state, [1, self.state_dim])
				action = np.reshape(action, [1, 1])
				next_state = np.reshape(next_state, [1, self.state_dim])
				reward = np.reshape(reward, [1, 1])
				log_old_policy = np.reshape(log_old_policy, [1, 1])

				state_batch.append(state)
				action_batch.append(action)
				reward_batch.append((reward + 8) / 8)
				old_policy_batch.append(log_old_policy)

				if len(state_batch) >= args.update_interval or done:
					states = self.list_to_batch(state_batch)
					actions = self.list_to_batch(action_batch)
					rewards = self.list_to_batch(reward_batch)
					old_policy = self.list_to_batch(old_policy_batch)

					v_values = self.critic.model.predict(states)
					next_v_value = self.critic.model.predict(next_state)

					gae, td_targets = self.gae_target(
						rewards, v_values, next_v_value, done)

					for epoch in range(args.epochs):
						actor_loss = self.actor.train(old_policy, states, actions, gae)
						critic_loss = self.critic.train(states, td_targets)

						with summary_writer.as_default():
							tf.summary.scalar("actor_loss: ", actor_loss, step=epoch)
							tf.summary.scalar("critic_loss: ", critic_loss, step=epoch)

					state_batch = []
					action_batch = []
					reward_batch = []
					old_policy_batch = []

				episodes_reward += reward[0][0]
				state = next_state[0]

			print('EP{} EpisodeReward={}'.format(epoch, episodes_reward))

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
