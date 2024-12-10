# -*- coding: utf-8 -*-

# Ranger 深度学习优化器 - RAdam + Lookahead + 校准自适应 LR 组合

import math
import torch
from torch.optim.optimizer import Optimizer

class RangerVA(Optimizer):
	"""
	Ranger deep learning optimizer - RAdam + Lookahead + 校准自适应 LR 组合
	"""

	def __init__(
			self,
			params,
			lr=1e-3,
			alpha=0.5,
			k=6,
			n_sma_threshold=5,
			betas=(0.95, 0.999),
			eps=1e-5,
			weight_decay=0,
			ams_grad=True,
			transformer="soft plus",
			smooth=50,
			grad_transformer="square"):
		"""
		初始化 Ranger 优化器
		:param params: 模型参数
		:param lr: 学习率
		:param alpha: Lookahead 参数
		:param k: Lookahead 参数
		:param n_sma_threshold: RangerAdam 参数
		:param betas: RangerAdam 参数
		:param eps: RangerAdam 参数
		:param weight_decay: 权重衰减
		:param ams_grad: RangerAdam 参数
		:param transformer: 学习率变换器类型
		:param smooth: 学习率变换器平滑参数
		:param grad_transformer: 梯度变换器类型
		"""
		# check parameters
		if not 0.0 <= alpha <= 1.0:
			raise ValueError(f'Invalid slow update rate: {alpha}')
		if not 1 <= k:
			raise ValueError(f'Invalid lookahead steps: {k}')
		if not lr > 0:
			raise ValueError(f'Invalid Learning Rate: {lr}')
		if not eps > 0:
			raise ValueError(f'Invalid eps: {eps}')

		# parameter comments:
		# beta1 (momentum) of .95 seems to work better than .90...
		# N_sma_threshold of 5 seems better in testing than 4.
		# In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

		# prep defaults and init torch.optim base
		defaults = dict(
			lr=lr,
			alpha=alpha,
			k=k,
			step_counter=0,
			betas=betas,
			n_sma_threshold=n_sma_threshold,
			eps=eps,
			weight_decay=weight_decay,
			smooth=smooth,
			transformer=transformer,
			grad_transformer=grad_transformer,
			ams_grad=ams_grad
		)
		super().__init__(params, defaults)
		# adjustable thresholds
		self.n_sma_threshold = n_sma_threshold
		# look ahead params
		self.k = k
		self.alpha = alpha
		# RAdam buffer for state
		self.rm_buffer = [[None, None, None] for ind in range(10)]

	def __setstate__(self, state):
		"""
		Restore the state of the optimizer
		:param state: Dictionary containing the state of the optimizer
		"""
		# print("set state called")
		super(RangerVA, self).__setstate__(state)

	def step(self, closure=None):
		"""
		Perform a single optimization step
		"""
		loss = None
		max_exp_avg_sq = 0.0
		# Evaluate averages and grad, update param tensors
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data.float()
				if grad.is_sparse:
					raise RuntimeError('Ranger optimizer does not support sparse gradients')
				ams_grad = group['ams_grad']
				smooth = group['smooth']
				grad_transformer = group['grad_transformer']
				p_data_fp32 = p.data.float()
				state = self.state[p]  # get state dict for this param
				if len(state) == 0:  # if first time to move state dict to CPU

					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(p_data_fp32)
					state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
					if ams_grad:
						# Maintains max of all exp. moving avg. of sq. grad. values
						state['max_exp_avg_sq'] = torch.zeros_like(p.data)
					# look ahead weight storage now in state dict
					state['slow_buffer'] = torch.empty_like(p.data)
					state['slow_buffer'].copy_(p.data)
				else:
					state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
					state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
				# begin computations
				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']
				if ams_grad:
					max_exp_avg_sq = state['max_exp_avg_sq']
				# compute variance mov avg
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
				# compute mean moving avg
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				# transformer
				grad_tmp = 0.0
				if grad_transformer == 'square':
					grad_tmp = grad ** 2
				elif grad_transformer == 'abs':
					grad_tmp = grad.abs()
				exp_avg_sq.mul_(beta2).add_((1 - beta2) * grad_tmp)
				if ams_grad:
					# Maintains the maximum of all 2nd moment running avg. till now
					torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
					# Use the max. for normalizing running avg. of gradient
					dynamic = max_exp_avg_sq.clone()
				else:
					dynamic = exp_avg_sq.clone()
				if grad_transformer == 'square':
					# pdb.set_trace()
					dynamic.sqrt_()
				state['step'] += 1
				if group['weight_decay'] != 0:
					p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']
				step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
				# ...let's use calibrated alr
				if group['transformer'] == 'soft plus':
					sp = torch.nn.Softplus(smooth)
					dynamics = sp(dynamic)
					p_data_fp32.addcdiv_(-step_size, exp_avg, dynamics)

				else:
					dynamic = exp_avg_sq.sqrt().add_(group['eps'])
					p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, dynamic)
				p.data.copy_(p_data_fp32)
				# integrated look ahead...
				# we do it at the param level instead of group level
				if state['step'] % group['k'] == 0:
					slow_p = state['slow_buffer']  # get access to slow param tensor
					slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
					p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor
		return loss