# -*- coding: utf-8 -*-

"""
todo: 在这里，包装了原始环境，以便于使用。当游戏结束时，不会手动重置环境，而是自动重置
"""

import torch
import numpy as np

ALL_ROLE = (
    'landlord1',
    'landlord2',
    'landlord3',
    'landlord4',
)


def _format_observation(obs, device):
    """
    用于处理观察并将其移动到CUDA的实用函数
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {
        'x_batch': x_batch,
        'z_batch': z_batch,
        'legal_actions': obs['legal_actions'],
    }
    return position, obs, x_no_action, z


class Environment:
    def __init__(self, env, device):
        """
        初始化此环境包装器
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset(), self.device)
        self.episode_return = {role: torch.zeros(1, 1) for role in ALL_ROLE}
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        # position = obs['position']
        # self.episode_return[position] += reward.get(position, 0.0)
        episode_return = self.episode_return

        if done:
            obs = self.env.reset()
            for p in ALL_ROLE:
                self.episode_return[p] += reward.get(p, 0.0)
            episode_return = self.episode_return
            self.episode_return = {role: torch.zeros(1, 1) for role in ALL_ROLE}

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        # reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )

    def close(self):
        self.env.close()