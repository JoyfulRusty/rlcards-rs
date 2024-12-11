# -*- coding: utf-8 -*-

import csv
import torch
from torch.backends import cudnn

import numpy as np

def cal_time(func, *args, **kwargs):
    def inner():
        import time
        s = time.time()
        func(*args, **kwargs)
        print("{}耗时：".format(func.__name__), time.time() - s)

    return inner

def set_seed(seed):
    """设置随机种子"""
    if seed is None:
        import subprocess
        import sys
        seed = 0

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_package = [r.decode().split('.')[0] for r in reqs.split()]

        if 'torch' in installed_package:
            cudnn.deterministic = True
            torch.manual_seed(seed)

        np.random.seed(seed)
        import random
        random.seed(seed)

def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--> Running on the GPU <--")
    else:
        device = torch.device("cpu")
        print("--> Running on thr CPU <--")

    return device

def reorganize(trajectories, payoffs):
    """
    重新组织轨迹，使其适合RL
    :param trajectories: 轨迹列表
    :param payoffs: 玩家的奖励列表， 每一条目对应一个玩家
    :return:
        (list): 一个新的轨迹，可以输入RL算法
    """
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player])-3, 2):
            if i == len(trajectories[player]) - 4:
                reward = payoffs[player]
                done = True
            else:
                reward, done = 0, False

            transition = trajectories[player][i:i + 3].copy()

            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)

    return new_trajectories


def remove_illegal(action_pr_obs, legal_actions):
    """
    删除非法操作，并将概率向量三处, 一维numpy数组, 合法行为为索引列表
    pr_obs(np.array): 没有合法行为的规范化向量
    """
    pr_obs = np.zeros(action_pr_obs.shape[0])
    pr_obs[legal_actions] = action_pr_obs[legal_actions]

    if np.sum(pr_obs) == 0:
        pr_obs[legal_actions] = 1 / len(legal_actions)

    else:
        pr_obs /= sum(pr_obs)

    return pr_obs


def tournament(env, num):
    """
    评估代理在环境中的表现
    :param env: 需要评估的环境
    :param num: 要玩的游戏数
    :return:
        每位玩家的平均奖金列表
    """
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:  # 1000
        _, _payoffs, _, _ = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter  # 奖励·计算 -> 用收益(也就是游戏赢的次数) / 统计的数量

    # 每个玩家收益除以对局数量，得到所有对局的平均收益值
    return payoffs

def plot_curve(csv_path, save_path, algorithm):
    """从csv文件读取数据并绘制结果"""
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['episode']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

def calc_single_card(curr_hand_cards):
    _list = list(curr_hand_cards)
    n = len(_list)
    if n <= 1:
        print(curr_hand_cards)
        return
    list1 = []
    list2 = []
    list3 = []
    for i in range(n - 1):
        if _list[i] != _list[i + 1]:
            list1.append(_list[i])
        if _list[i] == _list[i + 1]:
            list2.append(_list[i + 1])
            list1.append(_list[-1])
    str1 = ''.join(list1)
    str2 = ''.join(set(list2))
    for i in range(len(list(str2)) - 1):
        if list(str2)[i] != list(str2)[i + 1]:
            list3.append(list(str2)[i + 1])
    str3 = ''.join(list3)

    return str1, str2, str3

def read_data(csv_path):
    """ 读取玩家奖励数据 """
    xs = []
    ys = []
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            xs.append((int(row['player']), int(row['episode'])))
            ys.append((int(row['player']), float(row['reward'])))
    return xs, ys

def plot_curve_1(csv_path, save_path, i):
    """从csv文件读取数据并绘制结果"""
    import os
    import matplotlib.pyplot as plt
    xs, ys = read_data(csv_path)
    tmp_xs, tmp_ys = [], []
    # episode
    for x in xs:
        if x[0] == i:
            tmp_xs.append(x[1])
        continue
    # reward
    for y in ys:
        if y[0] == i:
            tmp_ys.append(y[1])
        continue
    fig, ax = plt.subplots()
    ax.plot(tmp_xs, tmp_ys, label='DQN' + '-' + str(i))
    ax.set(xlabel='episode' + '-' + str(i), ylabel='reward' + '-' + str(i))
    ax.legend()
    ax.grid()

    save_dir = os.path.dirname(save_path[i])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(save_path[i])

def begin_plot(csv_path, save_path):
    """ 开始绘图 """
    p_id = [0, 1, 2, 3]
    for i in p_id:
        plot_curve_1(csv_path, save_path, i)