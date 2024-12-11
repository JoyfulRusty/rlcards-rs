# -*- coding: utf-8 -*-

import torch
import typing
import logging
import traceback
import numpy as np

from collections import Counter
from rlcards.games.pig.env.env import Env
from rlcards.games.pig.env.gz import Environment
from rlcards.const.pig.const import Card2Column, NumOnes2Array

log_handle = logging.StreamHandler()
log_handle.setFormatter(
    logging.Formatter('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'))
log = logging.getLogger('pig')
log.propagate = False
log.addHandler(log_handle)
log.setLevel(logging.INFO)

# 缓冲区用于在参与者进程之间传输数据，以及学习过程。它们是GPU中的共享张量
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    """
    创建游戏环境
    """
    return Env(flags.objective)


def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    此函数将从基于基于从完整队列接收到的索引。
    它还将通过将索引发送到full_queue来释放索引。
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def create_optimizers(flags, learner_model):
    """
    对应位置创建对应的优化器
    """
    positions = ['landlord1', 'landlord2', 'landlord3', 'landlord4']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers


def create_buffers(flags, device_iterator):
    """
    为不同的位置以及用于不同的设备(即GPU)。
    也就是说，每个设备将具有用于三个位置的三个缓冲器。
    """
    T = flags.unroll_length
    positions = ['landlord1', 'landlord2', 'landlord3', 'landlord4']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            # x_dim = 319 if position == 'landlord' else 430
            x_dim = 780  # todo: shape改变
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 52), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 208), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    这个函数将永远运行，直到我们停止它。它将生成并将数据发送到缓冲区。
    它使用与主进程同步的空闲队列和完整队列
    """
    positions = ['landlord1', 'landlord2', 'landlord3', 'landlord4']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                position, obs, env_output = env.step(action)
                size[position] += 1  # 出牌次数？
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)

                            # episode_return = env_output['episode_return'] if p == 'landlord' else -env_output[
                            #     'episode_return']
                            episode_return = env_output['episode_return'][p]  # 每个玩家reward分别结算
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            # print("target_buf: ", [episode_return for _ in range(diff)])
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in positions:
                if size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def _cards2tensor_old(list_cards):
    """
    将整数列表转换为张量代表
    """
    if len(list_cards) == 0:
        return torch.zeros(54, dtype=torch.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    matrix = np.concatenate((matrix.flatten('F'), jokers))
    matrix = torch.from_numpy(matrix)
    return matrix


def _cards2tensor(list_cards):
    """
    one_hot编码
    return: shape:(1,52)
    """
    if len(list_cards) == 0:
        return torch.zeros(52, dtype=torch.int8)
    matrix = np.zeros([4, 13], dtype=np.int8)
    for c in list_cards:
        suit = int(c // 100)
        val = int(c % 100)
        val = val - 2 if suit != 3 else val - 4
        matrix[suit - 1, val] = 1
    # matrix = matrix.flatten('F')  # todo: F按竖的方向降，是否有问题？
    matrix = matrix.flatten()
    matrix = torch.from_numpy(matrix)
    return matrix


if __name__ == '__main__':
    lst = [3, 3, 3, 3, 20, 30]
    res = _cards2tensor(lst)
    print(res)
