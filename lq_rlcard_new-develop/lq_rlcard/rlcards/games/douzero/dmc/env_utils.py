# -*- coding: utf-8 -*-

import torch 

def _format_observation(obs, device):
    """
    处理观测和将它们转移到CUDA
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z

class Environment:
    def __init__(self, env, device):
        """
        初始化环境包装器
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        """
        环境参数初始化
        :return:
        """
        obs = self.env.reset()
        initial_position, initial_obs, x_no_action, z = _format_observation(obs, self.device)
        # initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z
        )

    def step(self, action):
        """
        更新动作
        """
        # env
        obs, reward, done, _ = self.env.step(action)

        # 奖励
        self.episode_return += reward
        episode_return = self.episode_return 

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)
        # 重新获取玩家state get_state
        position, obs, x_no_action, z = _format_observation(obs, self.device)
        # reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z
        )

    def close(self):
        self.env.close()