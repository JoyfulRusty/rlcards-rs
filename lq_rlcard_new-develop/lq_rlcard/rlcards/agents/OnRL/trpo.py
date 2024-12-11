# -*- coding: utf-8 -*-

# todo: T R P O算法

'''

'''

import gym
import copy
import torch
import rl_utils
import matplotlib.pyplot as plt
import torch.nn.functional as F

def compute_advantage(gamma, lm_bda, td_delta):
	"""
	计算TD差距
	"""
	td_delta = td_delta.detach().numpy()
	advantage_list = []
	advantage = 0.0
	for delta in td_delta[::-1]:
		advantage = gamma * lm_bda * advantage + delta
		advantage_list.append(advantage)
	advantage_list.reverse()
	return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
	"""
	策略网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化策略网络参数
		"""
		super(PolicyNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
	"""
	值网络
	"""
	def __init__(self, state_dim, hidden_dim):
		"""
		初始化网络参数
		"""
		super(ValueNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		return self.fc2(x)


class TRPO:
	"""
	T R P O
	"""
	def __init__(
			self,
			hidden_dim,
			state_space,
			action_space,
			lm_bda,
			kl_constraint,
			alpha,
			critic_lr,
			gamma,
			device):
		"""
		初始化 T R P O 模型训练参数
		"""
		state_dim = state_space.shape[0]
		action_dim = action_space.n

		# 策略网络不需要对优化器更新
		self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
		self.critic = ValueNet(state_dim, hidden_dim).to(device)

		# 优化器
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.gamma = gamma
		self.lm_bda = lm_bda  # gae参数
		self.alpha = alpha  # 线性搜索参数
		self.kl_constraint = kl_constraint  # KL距离最大限制
		self.device = device

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		action_prob = self.actor(state)
		action_dist = torch.distributions.Categorical(action_prob)
		action = action_dist.sample()
		return action.item()

	def hessian_matrix_vector_product(self, states, old_action_dists, vector):
		"""
		计算黑塞矩阵和一个向量的乘积
		"""
		new_action_dists = torch.distributions.Categorical(self.actor(states))
		# 计算平均KL距离
		kl = torch.mean(torch.distributions.kl_divergence(old_action_dists, new_action_dists))
		kl_grad = torch.autograd.grad(
			kl,
			self.actor.parameters(),
			create_graph=True
		)
		kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
		# KL距离的梯度先和向量进行点积运算
		kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
		grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
		grad2_vector = torch.cat([grad.view(-1) for grad in grad2])

		return grad2_vector

	def conjugate_gradient(self, grad, states, old_action_dists):
		"""
		共轭梯度法求解方程
		"""
		x = torch.zeros_like(grad)
		r = grad.clone()
		p = grad.clone()
		r_dot_r = torch.dot(r, r)
		# 共轭梯度主循环
		for i in range(10):
			hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
			alpha = r_dot_r / torch.dot(p, hp)
			x += alpha * p
			r -= alpha * p
			new_r_dot_r = torch.dot(r, r)
			if new_r_dot_r < 1e-10:
				break
			beta = new_r_dot_r / r_dot_r
			p = r + beta * p
			r_dot_r = new_r_dot_r
		return  x

	def compute_surrogate_obj(
			self,
			states,
			actions,
			advantage,
			old_log_prob,
			actor):
		"""
		计算策略目标
		"""
		log_prob = torch.log(actor(states).gather(1, actions))
		ratio = torch.exp(log_prob - old_log_prob)
		return torch.mean(ratio * advantage)

	def line_search(
			self,
			states,
			actions,
			advantage,
			old_log_prob,
			old_action_dists,
			max_vec):
		"""
		线性搜索
		"""
		old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
		old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, self.actor)
		# 线性搜索主循环
		for i in range(15):
			coe = self.alpha ** i
			new_para = old_para * coe * max_vec
			new_actor = copy.deepcopy(self.actor)
			torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
			new_action_dists = torch.distributions.Categorical(new_actor(states))
			kl_div = torch.mean(torch.distributions.kl_divergence(old_action_dists, new_action_dists))
			new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, new_actor)
			if new_obj > old_obj and kl_div < self.kl_constraint:
				return new_para
		return old_obj

	def policy_learn(
			self,
			states,
			actions,
			old_action_dists,
			old_log_prob,
			advantage):
		"""
		更新策略函数
		"""
		surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, self.actor)
		grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
		obj_grad = torch.cat([grad.view(-1) for grad in grads])
		# 用共轭梯度法计算x = H^(-1)g
		descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
		hp = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
		max_coe = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, hp) + 1e-8))
		# 线性搜索
		new_para = self.line_search(states, actions, advantage, old_log_prob, old_action_dists, descent_direction * max_coe)
		# 用线性搜索后的参数更新策略
		torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

	def update(self, transition_dict):
		"""
		更新模型参数
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		# TD差距
		td_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
		td_delta = td_target - self.critic(states)
		advantage = compute_advantage(self.gamma, self.lm_bda, td_delta.cpu()).to(self.device)
		old_log_prob = torch.log(self.actor(states).gather(1, actions)).detach()
		old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
		critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

		# 梯度归零
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# 更新价值函数
		self.critic_optimizer.step()
		# 更新策略
		self.policy_learn(states, actions, old_action_dists, old_log_prob, advantage)


def train_TRpo():
	"""
	训练模型
	"""
	num_episodes = 500
	hidden_dim = 128
	gamma = 0.98
	lm_bda = 0.95
	critic_lr = 1e-2
	kl_constraint = 0.0005
	alpha = 0.5
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# 构建环境和代理
	env_name = 'CartPole-v0'
	env = gym.make(env_name, render_mode="human")
	env.seed(0)
	torch.manual_seed(0)
	agent = TRPO(hidden_dim, env.observation_space, env.action_space, lm_bda,kl_constraint, alpha, critic_lr, gamma, device)
	return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('TRPO on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('TRPO on {}'.format(env_name))
	plt.show()


class PolicyNetContinuous(torch.nn.Module):
	"""
	连续策略网络
	"""
	def __init__(self, state_dim, hidden_dim, action_dim):
		"""
		初始化连续策略参数
		"""
		super(PolicyNetContinuous, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
		self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		"""
		前向传递
		"""
		x = F.relu(self.fc1(x))
		mu = 2.0 * torch.tanh(self.fc_mu(x))
		std = F.softplus(self.fc_std(x))

		return mu, std  # 高斯分布的均值和标准差


class TRPOContinuous:
	"""
	T R P O 处理连续动作的算法
	"""
	def __init__(
			self,
			hidden_dim,
			state_space,
			action_space,
			lm_bda,
			kl_constraint,
			alpha,
			critic_lr,
			gamma,
			device):
		"""
		初始化参数
		"""
		state_dim = state_space.shape[0]
		action_dim = action_space.shape[0]
		self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
		self.critic = ValueNet(state_dim, hidden_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.gamma = gamma
		self.lm_bda = lm_bda
		self.kl_constraint = kl_constraint
		self.device = device
		self.alpha = alpha

	def take_action(self, state):
		"""
		计算下一个动作
		"""
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		mu, std = self.actor(state)
		action_dist = torch.distributions.Normal(mu, std)
		action = action_dist.sample()
		return [action.item()]

	def hessian_matrix_vector_product(self, states, old_action_dists, vector, damping=0.1):
		"""
		矩阵向量相乘
		"""
		mu, std = self.actor
		new_action_dists = torch.distributions.Normal(mu, std)
		kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
		kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
		kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
		kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
		grad2 = torch.autograd.grad(kl_grad_vector_product,self.actor.parameters())
		grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
		return grad2_vector + damping * vector

	def conjugate_gradient(self, grad, states, old_action_dists):
		"""
		共轭基
		"""
		x = torch.zeros_like(grad)
		r = grad.clone()
		p = grad.clone()
		r_dot_r = torch.dot(r, r)
		# 共轭梯度主循环
		for i in range(10):
			hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
			alpha = r_dot_r / torch.dot(p, hp)
			x += alpha * p
			r -= alpha * p
			new_r_dot_r = torch.dot(r, r)
			if new_r_dot_r < 1e-10:
				break
			beta = new_r_dot_r / r_dot_r
			p = r + beta * p
			r_dot_r = new_r_dot_r
		return x

	def compute_surrogate_obj(
			self,
			states,
			actions,
			advantage,
			old_log_prob,
			actor):
		mu, std = actor(states)
		action_dists = torch.distributions.Normal(mu, std)
		log_prob = action_dists.log_prob(actions)
		ratio = torch.exp(log_prob - old_log_prob)
		return torch.mean(ratio * advantage)

	def line_search(
			self,
			states,
			actions,
			advantage,
			old_log_prob,
			old_action_dists,
			max_vec):
		"""
		线性搜索
		"""
		old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
		old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, self.actor)
		# 线性搜索主循环
		for i in range(15):
			coe = self.alpha ** i
			new_para = old_para * coe * max_vec
			new_actor = copy.deepcopy(self.actor)
			torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
			new_action_dists = torch.distributions.Categorical(new_actor(states))
			kl_div = torch.mean(torch.distributions.kl_divergence(old_action_dists, new_action_dists))
			new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, new_actor)
			if new_obj > old_obj and kl_div < self.kl_constraint:
				return new_para
		return old_obj

	def policy_learn(
			self,
			states,
			actions,
			old_action_dists,
			old_log_prob,
			advantage):
		"""
		更新策略函数
		"""
		surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, self.actor)
		grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
		obj_grad = torch.cat([grad.view(-1) for grad in grads])
		# 用共轭梯度法计算x = H^(-1)g
		descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
		hp = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
		max_coe = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, hp) + 1e-8))
		# 线性搜索
		new_para = self.line_search(states, actions, advantage, old_log_prob, old_action_dists, descent_direction * max_coe)
		# 用线性搜索后的参数更新策略
		torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

	def update(self, transition_dict):
		"""
		更新模型参数
		"""
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		# TD差距
		td_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
		td_delta = td_target - self.critic(states)
		advantage = compute_advantage(self.gamma, self.lm_bda, td_delta.cpu()).to(self.device)
		old_log_prob = torch.log(self.actor(states).gather(1, actions)).detach()
		old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
		critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

		# 梯度归零
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# 更新价值函数
		self.critic_optimizer.step()
		# 更新策略
		self.policy_learn(states, actions, old_action_dists, old_log_prob, advantage)

def train_TRpoContinuous():
	"""
	训练连续动作
	"""
	num_episodes = 2000
	hidden_dim = 128
	gamma = 0.9
	lm_bda = 0.9
	critic_lr = 1e-2
	kl_constraint = 0.00005
	alpha = 0.5
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# 构建训练环境
	env_name = 'Pendulum-v1'
	env = gym.make(env_name)
	env.seed(0)
	torch.manual_seed(0)
	agent = TRPOContinuous(
		hidden_dim,
		env.observation_space,
		env.action_space,
		lm_bda,
		kl_constraint,
		alpha,
		critic_lr,
		gamma,
		device
	)

	return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('TRPO on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('TRPO on {}'.format(env_name))
	plt.show()






if __name__ == '__main__':
	# train_TRpo()
    train_TRpoContinuous()