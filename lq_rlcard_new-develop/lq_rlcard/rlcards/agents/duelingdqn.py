# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy


# 动作过渡和转变
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])

class DuelDQNAgent(object):
    """DuelDQNAgent(DDQNAgent代理)"""
    def __init__(self,
                 replay_memory_size = 20000,
                 replay_memory_init_size = 100,
                 update_target_estimator_every = 10,
                 discount_factor = 0.99,
                 epsilon_start = 1.0,
                 epsilon_end = 0.1,
                 epsilon_decay_steps = 20000,
                 batch_size = 32,
                 num_actions = 2,
                 state_shape = None,
                 train_every = 1,
                 learning_rate = 0.00005,
                 device = None):
        """
        使用函数逼近的非策略TD控制Q学习算法，在遵循epsilon贪婪策略的同时找到最优贪婪策略
        :param replay_memory_size(int): 重放内存的大小
        :param replay_memory_init_size(int): 初始化时要采样的随机体验数回复存储器
        :param update_target_estimator_every(int): 将参数从Q估计器复制到每N步目标估计器
        :param discount_factor(float): 伽马折扣系数
        :param epsilon_start(float): 在执行动作时，对随机动作进行采样的机会，Epsilon随时间衰减，这是开始值
        :param epsilon_end(float): 衰减完成后，epsilon的最终最小值
        :param epsilon_decay_steps(int): 衰减epsilon的步数
        :param batch_size(int): 要重放内存中采样的批的大小
        :param evaluate_every(int): 每N步求值一次
        :param num_actions(int): 操作数
        :param state_space(int): 状态向量空间
        :param state_shape(int): 状态维度
        :param train_every(int): 每X步，训练一次网络
        :param mlp_layers(list): mlp中每个层的层号和维度
        :param learning_rate(float): DDQNAgent代理学习率
        :param device: CPU OR GPU
        """

        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        # 运行设备(GPU/CPU)
        if device is None:
            self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 总的时间步长
        self.total_t = 0

        # 总的训练步数
        self.train_t = 0

        # 衰减因子
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # 创建估计器
        self.q_estimator = DuelEstimator(num_actions = num_actions,
                                         learning_rate = learning_rate,
                                         state_shape = state_shape,
                                         device = self.device)

        # 创建目标估计器
        self.target_estimator = DuelEstimator(num_actions=num_actions,
                                              learning_rate = learning_rate,
                                              state_shape = state_shape,
                                              device = self.device)


        # 创建回放/重放记忆
        self.memory = Memory(replay_memory_size, batch_size)


    def feed(self, ts):
        """
        将数据存储到重放缓冲区并训练代理，两个阶段：
        1.在第一阶段，在没有训练的情况下填充内存
        2.在第二阶段，每隔几个时间步，训练一次代理
        :param ts(list): 表示转换的5个元素的列表
        """

        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(state['legal_actions'].keys()), done)
        self.total_t += 1

        # 创建临时变量
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()


    def step(self, state):
        """
        预测生成训练数据的操作，但将预测与计算图进行分开
        :param state(np.array): 当前状态
        :return:
            action(int): 操作id
        """
        q_values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        legal_actions = list(state['legal_actions'].keys())

        # 测验
        probs = np.ones(len(legal_actions), dtype = float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(q_values))

        # 测验最好的行为idx
        probs[best_action_idx] += (1.0 - epsilon)

        # 随机选择action_idx
        action_idx = np.random.choice(np.arange(len(probs)), p = probs)

        return legal_actions[action_idx]


    def eval_step(self, state):
        """
        预测用于评估的操作
        :param state(np.array): 当前状态
        :return:
            action(int): 操作id
            info(dict): 包含信息的字典
        """
        q_values = self.predict(state)
        best_action = np.argmax(q_values) # 获取q_values的最大值，来作best_action

        info = {}
        info['values'] = {
            state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]])
            for i in range(len(state['legal_actions']))
        }

        return best_action, info


    def predict(self, state):
        """
        预测掩蔽的q_values
        :param state(np.array): 当前的状态
        :return:
            q_values(np.array): 一个一维数组，其中每个条目代表一个q_values
        """

        q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]

        # 掩蔽的q_values
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype = float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values


    def train(self):
        """训练"""

        # 状态批次、动作批次、奖励批次、下一个动作批次、合法动作批次、是否done批次
        state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch = self.memory.sample()
        # print("next_state_batch: ", next_state_batch)
        # print("next_state_batch_shape: ", next_state_batch.shape)

        # 使用Q-network（双DQN）计算最佳下一步行动
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        legal_actions = []

        # 循环批次大小
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])

        # 掩蔽的q_values
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype = float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))

        # 获取masked_q_values中 best_action
        best_actions = np.argmax(masked_q_values, axis = 1)

        # 使用目标网络评估下一步最佳行动(DDQN)
        # q_values_next_target 下一个目标值
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)

        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                       self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]


        # 执行梯度下降更新, 将状态批次转换为np.array()
        state_batch = np.array(state_batch)

        # 损失函数
        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        if self.train_t % 50 == 0:
            print("INFO-Step {}, rl_loss: {}".format(self.total_t, loss))

        # 更新目标估计器
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            # print("\nINFO - Copied models parameters to target network.")

        self.train_t += 1

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        """
        向内存馈送进行转换
        :param state(np.array): 当前状态
        :param action(int): 执行的操作id
        :param reward(float): 收到的奖励
        :param next_state(np.array): 执行操作后的下一个状态
        :param legal_actions(list): 下一个合法行为
        :param done(boolean): 是否结束
        """
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        """设置device"""
        self.device = device
        self.q_estimator.device = device
        self.target_estimator.device = device

class Memory(object):
    """
    用于保存转换的内存
    """
    def __init__(self, memory_size, batch_size):
        """
        初始化参数
        :param memory_size(int): 缓存内存大小
        :param batch_size(int): 批次大小
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        """
        将转换保存到内存中
        :param state(np.array): 当前状态
        :param action(int): 执行操作的id
        :param reward(float): 收到的奖励
        :param next_state(bp.array): 执行操作后的后的下一个动作
        :param legal_actions(list): 下一个合法动作
        :param done(boolean): 是否结束
        """
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)

        transition = Transition(state, action, reward, next_state, legal_actions, done)
        self.memory.append(transition)


    def sample(self):
        """
        从回访记忆中采样一个才批次

        :param state_batch(list): 一批状态
        :param action_batch(list): 一批操作
        :param reward_batch(list): 一批奖励
        :param next_state_batch(list): 一批状态
        :param done_batch(list): 一批done
        """
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

class DUELingNet(nn.Module):
    """DUEL网络"""

    def __init__(self, state_size, n_actions):
        """
        初始化参数
        :param state_size(int): 状态批次大小
        :param n_actions(list): n个actions
        """
        super(DUELingNet, self).__init__()
        self.state_size = 1
        for s_size in state_size:
            self.state_size *= s_size
        self.n_actions = n_actions

        # 共享网络层
        self.fc1 = nn.Linear(self.state_size, self.state_size * 12) # (1, 10)
        self.fc2 = nn.Linear(self.state_size * 12, self.state_size * 6) # (10, 5)

        # 独享网络层
        self.fc3_adv = nn.Linear(self.state_size * 6, n_actions * 4) # (5， 4)
        self.fc3_val = nn.Linear(self.state_size * 6, n_actions * 4) # (5， 4)
        self.fc4_adv = nn.Linear(n_actions * 4, n_actions)
        self.fc4_val = nn.Linear(n_actions * 4, 1)

        # 设置激活层
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        前向传播
        :param x: 传入的值x
        """

        # 碾平
        x = torch.flatten(x, start_dim = 1)
        # x = torch.unsqueeze(x, dim=0)

        # 获取批次大小
        batch_size = x.shape[0]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        adv = self.fc3_adv(x)
        adv = self.relu(adv)
        val = self.fc3_val(x)
        val = self.relu(val)
        adv = self.fc4_adv(adv)
        val = self.fc4_val(val)
        adv = self.relu(adv)

        return val + adv - adv.mean(1).unsqueeze(1).expand(batch_size, self.n_actions) # unsqueeze 变形，切片


class DUELingConvNet(nn.Module):
    """
    DUELingConvNet 卷积网络
    """

    def __init__(self, state_size, n_actions): # state_size = [6, 34, 4], n_actions = 38
        """
        初始化参数
        :param state_size: 状态大小
        :param n_actions: n 个 actions
        """
        super(DUELingConvNet, self).__init__()

        self.state_size = 1 # 6 x 34 x 4
        for s_size in state_size: # [6, 18, 4]
            self.state_size *= s_size

        # actions
        self.n_actions = n_actions

        # 共享网络层
        self.conv1 = nn.Conv2d(state_size[0], 1, kernel_size = (1, 4), stride = (1, 1))
        self.conv2 = nn.Conv2d(state_size[0], 1, kernel_size = (2, 4), stride = (1, 1))
        self.conv3 = nn.Conv2d(state_size[0], 1, kernel_size = (3, 4), stride = (1, 1))
        self.conv4 = nn.Conv2d(state_size[0], 1, kernel_size = (4, 4), stride = (1, 1))

        # 独享网络层
        self.fc3_adv = nn.Linear(70, n_actions * 2)
        self.fc3_val = nn.Linear(70, n_actions * 2)
        self.fc4_adv = nn.Linear(n_actions * 2, n_actions)
        self.fc4_val = nn.Linear(n_actions * 2, 1)

        # 设置激活函数
        self.relu = nn.ReLU()

    def forward(self, x):

        output_1 = torch.flatten(self.relu(self.conv1(x)), 1)
        output_2 = torch.flatten(self.relu(self.conv2(x)), 1)
        output_3 = torch.flatten(self.relu(self.conv3(x)), 1)
        output_4 = torch.flatten(self.relu(self.conv4(x)), 1)

        x = torch.cat([output_1, output_2, output_3, output_4], 1)

        batch_size = x.shape[0]
        adv = self.fc3_adv(x)
        adv = self.relu(adv)
        val = self.fc3_val(x)
        val = self.relu(val)
        adv = self.fc4_adv(adv)
        val = self.fc4_val(val)
        return val + adv - adv.mean(1).unsqueeze(1).expand(batch_size, self.n_actions)

class DuelEstimator(object):

    def __init__(self, num_actions = 2, learning_rate = 0.001, state_shape = None, device = None):
        """
        初始化Estimator(估计器)对象
        :param num_actions(int): 出书操作数
        :param learning_rate(float): 学习效率
        :param state_shape(list): 状态空间的形状
        :param device: CPU OR GPU
        """
        self.num_actions = num_actions # 38
        self.learning_rate = learning_rate
        self.state_shape = state_shape # [6, 34, 4]
        self.device = device

        # 设置Q模型并将其置于eval模式
        qnet = DUELingConvNet(state_shape, num_actions)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # 使用Xavier init初始化权重
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # 设置损失函数
        self.mse_loss = nn.MSELoss(reduction = 'mean')

        # 设置优化器
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr = self.learning_rate)

    def predict_nograd(self, s):
        """
        预测操作值，但不包括预测在计算图中。用于预测最佳的下一个DDQN算法操作
        :param s(np.array): (批处理， state_len)
        :return:
            np.ndarray(shape): (batch_size, NUM_VALID_ACTIONS), 包含估计的动作值
        """
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        """
        向给定目标更新估计器，在这种情况下，y是估计的目标网络，Q网络最佳行动的值
        :param s(np.array): (batch_size, state_shape)，状态表示
        :param a(np.array): (batch_size, ) 整数采样操作
        :param y(np.array): 根据Q目标的最佳行为的(批次)的值
        :return:
            batch_size: 批次的计算损失
        """
        self.optimizer.zero_grad()
        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # 输入状态获得所有动作的价值，(batch, state_shape) -> (batch, num_actions)
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )

        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # 更新model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss