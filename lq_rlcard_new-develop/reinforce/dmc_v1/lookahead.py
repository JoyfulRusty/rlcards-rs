# -*- coding: utf-8 -*-

import torch
from torch.optim import Optimizer
from collections import defaultdict

class LookAhead(Optimizer):
    """
    优化器
    """
    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        """
        :param optimizer: 内部优化器
        :param k (int): 步骤数
        :param alpha(float): 线性插值因子, 1.0恢复了内部优化器
        :param pullback_momentum (str): 插值更新时内部优化器动量的变化
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # 缓存当前优化器参数
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self, **kwargs):
        """
        梯度归零
        """
        self.optimizer.zero_grad()

    def state_dict(self):
        """
        状态字典
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        加载状态字典
        """
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """
        有助于对慢速权重进行评估(通常可以更好地概括)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        """
        清除并加载备份
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        """
        执行单个优化步骤
        closure: 重新评估模型的闭包并返回损失
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # 前瞻并缓存当前优化器参数
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    # 关键线路
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state['cached_params'])
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).addl_(
                            1.0 - self.alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss