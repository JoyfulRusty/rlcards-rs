# -*- coding: utf-8 -*-

import numpy as np


class RandomAgent(object):
    """
    todo: 随机代理
    """

    def __init__(self, num_actions):
        """
        初始化动作参数
        """
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        """
        迭代动作
        """
        return np.random.choice(list(state['actions'].keys()))

    def eval_step(self, state):
        """
        校验当前动作价值
        """
        prob = [0 for _ in range(self.num_actions)]
        for i in state['actions']:
            prob[i] = 1/len(state['actions'])

        info = {}

        for i in range(len(state['actions'])):
            value = prob[list(state['actions'].keys())[i]]
            info['prob'] = {state['row_legal_actions'][i]: value}

        return self.step(state), info