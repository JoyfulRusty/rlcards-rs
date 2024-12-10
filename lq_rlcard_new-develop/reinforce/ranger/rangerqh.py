# -*- coding: utf-8 -*-

# 将准双曲动量与 Hinton Lookahead 相结合

import torch
from torch.optim.optimizer import Optimizer


class RangerQH(Optimizer):
	"""
	RangerQH optimizer, a combination of Ranger and Hinton Lookahead.
	"""

	def __init__(
			self,
			params,
			lr=1e-3,
			betas=(0.9, 0.999),
			nus=(.7, 1.0),
			weight_decay=0.0,
			k=6,
			alpha=.5,
			decouple_weight_decay=False,
			eps=1e-8):
		"""
		初始化 RangerQH 优化器
		:param params: 模型参数
		:param lr: 学习率
		:param betas: 动量参数
		:param nus: 二次动量参数
		:param weight_decay: 权重衰减
		:param k: Lookahead 步数
		:param alpha: Lookahead 步长
		:param decouple_weight_decay: 是否解耦权重衰减
		:param eps: 数值稳定性
		"""
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = {
			"lr": lr,
			"betas": betas,
			"nus": nus,
			"weight_decay": weight_decay,
			"decouple_weight_decay": decouple_weight_decay,
			"eps": eps,
		}
		super().__init__(params, defaults)

		# look ahead params
		self.alpha = alpha
		self.k = k

	def step(self, closure=None):
		"""
		执行单个优化步骤
		重新评估模型并返回损失的闭包
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			lr = group["lr"]
			beta1, beta2 = group["betas"]
			nu1, nu2 = group["nus"]
			weight_decay = group["weight_decay"]
			decouple_weight_decay = group["decouple_weight_decay"]
			eps = group["eps"]

			for p in group["params"]:
				if p.grad is None:
					continue

				d_p = p.grad.data
				if d_p.is_sparse:
					raise RuntimeError("QHAdam does not support sparse gradients")

				if weight_decay != 0:
					if decouple_weight_decay:
						p.data.mul_(1 - lr * weight_decay)
					else:
						d_p.add_(weight_decay, p.data)

				d_p_sq = d_p.mul(d_p)

				# prep for saved param loading
				param_state = self.state[p]

				if len(param_state) == 0:
					param_state["beta1_weight"] = 0.0
					param_state["beta2_weight"] = 0.0
					param_state['step'] = 0
					param_state["exp_avg"] = torch.zeros_like(p.data)
					param_state["exp_avg_sq"] = torch.zeros_like(p.data)
					# look ahead weight storage now in state dict
					param_state['slow_buffer'] = torch.empty_like(p.data)
					param_state['slow_buffer'].copy_(p.data)

				param_state['step'] += 1

				param_state["beta1_weight"] = 1.0 + beta1 * param_state["beta1_weight"]
				param_state["beta2_weight"] = 1.0 + beta2 * param_state["beta2_weight"]

				beta1_weight = param_state["beta1_weight"]
				beta2_weight = param_state["beta2_weight"]
				exp_avg = param_state["exp_avg"]
				exp_avg_sq = param_state["exp_avg_sq"]

				beta1_adj = 1.0 - (1.0 / beta1_weight)
				beta2_adj = 1.0 - (1.0 / beta2_weight)
				exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
				exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)

				avg_grad = exp_avg.mul(nu1)
				if nu1 != 1.0:
					avg_grad.add_(1.0 - nu1, d_p)

				avg_grad_rms = exp_avg_sq.mul(nu2)
				if nu2 != 1.0:
					avg_grad_rms.add_(1.0 - nu2, d_p_sq)
				avg_grad_rms.sqrt_()
				if eps != 0.0:
					avg_grad_rms.add_(eps)

				p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)

				# integrated look ahead...
				# we do it at the param level instead of group level
				if param_state['step'] % self.k == 0:  # group['k'] == 0:
					slow_p = param_state['slow_buffer']  # get access to slow param tensor
					slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
					p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

		return loss

	@classmethod
	def _params_to_dict(cls, params):
		return {"lr": params.alpha, "nus": (params.nu1, params.nu2), "betas": (params.beta1, params.beta2)}