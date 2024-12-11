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
	def __init__(self, state_dim, action_dim):
		"""
		初始化参数
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim

		# 构建模型、优化器
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.actor_lr)

	def create_model(self):
		"""
		构建神经网络层
		"""
		actor_model = tf.keras.Sequential([
			tf.keras.layers.Input((self.state_dim,)),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(16, activation='relu'),
			tf.keras.layers.Dense(self.action_dim, activation='softmax')
		])

		return actor_model

	@staticmethod
	def compute_loss(old_policy, new_policy, actions, gae):
		"""
		计算损失值
		"""
		gae = tf.stop_gradient(gae)
		old_log_p = tf.math.log(tf.reduce_sum(old_policy * actions))
		old_log_p = tf.stop_gradient(old_log_p)
		log_p = tf.math.log(tf.reduce_sum(new_policy * actions))
		ratio = tf.math.exp(log_p - old_log_p)
		clipped_ratio = tf.clip_by_value(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
		surrogate = -tf.minimum(ratio * gae, clipped_ratio * gae)

		return tf.reduce_mean(surrogate)

	def train(self, old_policy, states, actions, gae):
		"""
		训练
		"""
		actions = tf.one_hot(actions, self.action_dim)
		actions = tf.reshape(actions, [-1, self.action_dim])
		actions = tf.cast(actions, tf.float64)

		with tf.GradientTape() as tape:
			p_value = self.model(states, training=True)
			loss = self.compute_loss(old_policy, p_value, actions, gae)

		grads = tape.gradient(loss, self.model.trainable_variables)
		self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

		return loss


class Critic:
	"""
	Critic评价代理
	"""
	def __init__(self, state_dim):
		"""
		初始化参数
		"""
		self.state_dim = state_dim
		self.model = self.create_model()
		self.opt = tf.keras.optimizers.Adam(args.critic_lr)

	def create_model(self):
		"""
		构建神经网络层
		"""
		critic_model = tf.keras.Sequential([
			tf.keras.layers.Input((self.state_dim,)),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(16, activation='relu'),
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
		初始化参数
		"""
		self.env = env
		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.n

		# 构建模型
		self.actor = Actor(self.state_dim, self.action_dim)
		self.critic = Critic(self.state_dim)

	@staticmethod
	def gae_target(rewards, v_values, next_v_value, done):
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
		启动训练
		"""
		for epoch in range(max_episodes):
			state_batch = []
			action_batch = []
			reward_batch = []
			old_policy_batch = []
			episode_reward, done = 0, False
			state = self.env.reset()[0]

			while not done:
				action_prob = self.actor.model.predict(np.reshape(state, [1, self.state_dim]))
				action = np.random.choice(self.action_dim, p=action_prob[0])

				next_state, reward, done, _, _ = self.env.step(action)
				state = np.reshape(state, [1, self.state_dim])
				action = np.reshape(action, [1, 1])
				next_state = np.reshape(next_state, [1, self.state_dim])
				reward = np.reshape(reward, [1, 1])

				state_batch.append(state)
				action_batch.append(action)
				reward_batch.append(reward * 0.01)
				old_policy_batch.append(action_prob)

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

				episode_reward += reward[0][0]
				state = next_state[0]

			print('EP{} EpisodeReward={}'.format(epoch, episode_reward))


def main():
	"""
	启动
	"""
	env_name = 'CartPole-v1'
	# 创建环境
	env = gym.make(env_name, render_mode="human")
	agent = Agent(env)
	agent.train()

if __name__ == "__main__":
	main()
