# -*- coding: utf-8 -*-
import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcards.agents.dqn_agent_zh import DQNAgent
from rlcards.utils.utils import remove_illegal

Transition = collections.namedtuple('Transition', 'info_state action_prob')

class NFSPAgent(object):
    """
    近似克隆，使用pytorch框架，而不是tensorflow框架
    此实现与Henrich和Silver(2016)的不同之处在于训练使交叉熵最小化行动概率而不是已实现的行动
    """
    def __init__(self,
                 num_actions=4,
                 state_shape=None,
                 action_shape=None,
                 hidden_layers_sizes=None,
                 reservoir_buffer_capacity=20000,
                 anticipatory_param=0.1,
                 batch_size=256,
                 train_every=1,
                 rl_learning_rate=0.1,
                 sl_learning_rate=0.005,
                 min_buffer_size_to_learn=100,
                 q_replay_memory_size=20000,
                 q_replay_memory_init_size=100,
                 q_update_target_estimator_every=1000,
                 q_discount_factor=0.99,
                 q_epsilon_start=0.06,
                 q_epsilon_end=0,
                 q_epsilon_decay_steps=int(1e6),
                 q_batch_size=32,
                 q_train_every=1,
                 q_mlp_layers=None,
                 evaluate_with='average_policy',
                 device=None,
                 save_path=None,
                 save_every=float('inf')):
        """
        TODO: 初始化代理参数
        num_actions(int): 动作次数
        state_shape(list): 状态空间的形状
        hidden_layers_sizes(list): 平均策略层的隐藏层大小
        reservoir_buffer_capacity(int): 平均策略的缓冲大小
        anticipatory_param(float): 平衡 rl/average策略的超参数
        batch_size(int): 用于训练平均策略的batch_size
        train_every(int): 每X步训练SL策略
        rl_learning_rate(float): RL代理的学习率
        sl_learning_rate(float): 平均策略的学习率
        min_buffer_size_to_learn(int): 平均策略学习的最小缓冲区大小
        q_replay_memory_size(int): 内部DQN代理的内存大小
        q_replay_memory_init_size(int): 内部DQN代理的初始内存大小
        q_update_target_estimator_every(int): 内部DQN代理更新目标网络的频率
        q_discount_factor(float): 内部DQN代理的折扣因子
        q_epsilon_start(float): 内部DQN代理的起始epsilon
        q_epsilon_end(float): 内部DQN代理的结束epsilon
        q_epsilon_decay_steps(int): 内部DQN代理的衰减步骤
        q_batch_size(int): 内部DQN代理的批量大小
        q_train_step(int): 每X步训练一次模型
        q_mlp_layers(list): 内部DQN代理的层大小
        device(torch.device): 是否使用cpu或gpu
        """
        self.use_raw = False
        self._num_actions = num_actions
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._layer_sizes = hidden_layers_sizes + [num_actions]
        self._batch_size = batch_size
        self._train_every = train_every
        self._sl_learning_rate = sl_learning_rate
        self._anticipatory_param = anticipatory_param
        self._min_buffer_size_to_learn = min_buffer_size_to_learn

        # 存储缓冲器，允许对数据流进行统一的采样
        self._prev_action = None
        self._prev_time_step = None
        self.evaluate_with = evaluate_with
        self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)

        # 判断cuda是否可用，否则就使用cpu
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 总时间步数
        self.total_t = 0

        # 计步器以跟踪学习情况
        self._step_counter = 0

        # 实例化DQN网络，用于学习
        # 构建行动价值网络(DQN)
        self._rl_agent = DQNAgent(
            q_replay_memory_size,
            q_replay_memory_init_size,
            q_update_target_estimator_every,
            q_discount_factor,
            q_epsilon_start,
            q_epsilon_end,
            q_epsilon_decay_steps,
            q_batch_size,
            num_actions,
            state_shape,
            q_train_every,
            q_mlp_layers,
            rl_learning_rate,
            device
        )

        # 构建平均政策网络模型
        self._build_model()

        # 采样批次策略
        self.sample_episode_policy()

    def _build_model(self):
        """
        构建平均政策网络模型
        """
        # 配置平均策略网络
        policy_network = AveragePolicyNetwork(self._state_shape, self._action_shape,)
        self.policy_network = policy_network
        self.policy_network.eval()

        # 泽维尔(xavier)初始化
        # xavier_uniform初始化是一种常用的权重初始化方法:
        #   1.旨在解决深度神经网络训练过程中的梯度消失和梯度爆炸问题
        #   2.该方法通过根据网络的输入和输出维度来初始化权重，使得前向传播和反向传播过程中信号保持一致的方差
        #   3.在xavier初始化中，权重的初始化范围由输入和输出维度共同决定
        #   4.xavier初始化通过从均匀分布中抽取权重值，使得权重方差等于输入和输出维度之和的倒数，这样可变卖你信号在网络中过度衰减或放大
        for p in self.policy_network.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # 优化器(Adam)
        # 较大的值(如 0.3)在学习率更新前会有更快的初始学习，而较小的值(如 1.0E-5)会令训练收敛到更好的性能
        self.policy_network_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self._sl_learning_rate
        )

    def feed(self, ts):
        """
        将数据馈送到内部RL代理，表示转换的5个元素的列表
        """
        self._rl_agent.feed(ts)
        self.total_t += 1
        if self.total_t > 0 and \
                len(self._reservoir_buffer) >= self._min_buffer_size_to_learn and \
                self.total_t % self._train_every == 0:
            sl_loss = self.train_sl()
            print('\rINFO - Step {}, sl-loss: {}'.format(self.total_t, sl_loss), end='')

    def step(self, state):
        """
        返回要采取的操作，更新状态和动作ID
        """
        action = []
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())

        # TODO: 不同策略
        # 最好的反映
        if self._mode == 'best_response':
            action = self._rl_agent.step(state)
            one_hot = np.zeros(self._num_actions)
            one_hot[action] = 1

            # 将新的过渡添加到存储缓冲区, 转换的形式为(状态，概率)
            self._add_transition(obs, one_hot)

        # 平均策略
        elif self._mode == 'average_policy':
            prob = self._act(obs)
            prob = remove_illegal(prob, legal_actions)
            action = np.random.choice(len(prob), p=prob)

        return action

    def eval_step(self, state):
        """
        使用平均策略进行评估，更新状态和动作ID
        """
        # todo: 最佳反映
        if self.evaluate_with == 'best_response':
            action, info = self._rl_agent.eval_step(state)

        # todo: 平均策略
        elif self.evaluate_with == 'average_policy':
            obs = state['obs']
            legal_actions = list(state['legal_actions'].keys())
            prob = self._act(obs)
            prob = remove_illegal(prob, legal_actions)
            action = np.random.choice(len(prob), p=prob)
            info = dict()
            info['prob'] = {
                state['raw_legal_actions'][i]: float(prob[list(state['legal_actions'].keys())[i]])
                for i in range(len(state['legal_actions']))}
        else:
            raise ValueError("'evaluate_with' should be either 'average_policy' or 'best_response'.")

        return action, info

    def sample_episode_policy(self):
        """
        平均/最佳响应策略
        """
        # todo: 选取策略，最佳和平均
        if np.random.rand() < self._anticipatory_param:
            self._mode = 'best_response'
        else:
            self._mode = 'average_policy'

    def _act(self, info_state):
        """
        根据观察结果和法律行动预测行动概率未连接到计算图，观察状态信息和动作转移概率
        """
        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(self.device)

        # 取消对梯度的计算
        with torch.no_grad():
            # 平均策略网络
            log_action_prob = self.policy_network(info_state).cpu().numpy()

        action_prob = np.exp(log_action_prob)[0]

        return action_prob

    def _add_transition(self, state, prob):
        """
        将新的过渡添加到存储缓冲区, 转换的形式为(状态，概率)
        """
        transition = Transition(info_state=state, action_prob=prob)
        self._reservoir_buffer.add(transition)

    def train_sl(self):
        """
        计算采样转换的损失并执行平均网络更新
        如果缓冲区中没有足够的元素，则不会计算损失，而是返回"None"，这批转换获得的平均损失或"无"
        """
        # 存储缓冲器中数据与批次大小
        if (len(self._reservoir_buffer) < self._batch_size or
                len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
            return None

        # 从存储缓冲器中采样批次大小的数据
        transitions = self._reservoir_buffer.sample(self._batch_size)
        info_states = [t.info_state for t in transitions]
        action_prob = [t.action_probs for t in transitions]

        # 取消对梯度的计算
        self.policy_network_optimizer.zero_grad()
        # 启动模型训练
        self.policy_network.train()

        # (batch, state_size)
        info_states = torch.from_numpy(np.array(info_states)).float().to(self.device)

        # (batch, num_actions)
        eval_action_prob = torch.from_numpy(np.array(action_prob)).float().to(self.device)

        # 平均策略网络
        # (batch, num_actions)
        log_forecast_action_prob = self.policy_network(info_states)

        ce_loss = - (eval_action_prob * log_forecast_action_prob).sum(dim=-1).mean()
        ce_loss.backward()

        self.policy_network_optimizer.step()
        ce_loss = ce_loss.item()
        self.policy_network.eval()

        return ce_loss

    def set_device(self, device):
        """
        设置训练设备
        """
        self.device = device
        self._rl_agent.set_device(device)

    def checkpoint_attributes(self):
        '''
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        '''

        return {
            'agent_type': 'NFSPAgent',
            'policy_network': self.policy_network.checkpoint_attributes(),
            'reservoir_buffer': self._reservoir_buffer.checkpoint_attributes(),
            'rl_agent': self._rl_agent.checkpoint_attributes(),
            'policy_network_optimizer': self.policy_network_optimizer.state_dict(),
            'device': self.device,
            'anticipatory_param': self._anticipatory_param,
            'batch_size': self._batch_size,
            'min_buffer_size_to_learn': self._min_buffer_size_to_learn,
            'num_actions': self._num_actions,
            'mode': self._mode,
            'evaluate_with': self.evaluate_with,
            'total_t': self.total_t,
            'train_t': self.train_t,
            'sl_learning_rate': self._sl_learning_rate,
            'train_every': self._train_every,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restore the model from a checkpoint

        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        '''
        print("\nINFO - Restoring model from checkpoint...")
        agent = cls(
            anticipatory_param=checkpoint['anticipatory_param'],
            batch_size=checkpoint['batch_size'],
            min_buffer_size_to_learn=checkpoint['min_buffer_size_to_learn'],
            num_actions=checkpoint['num_actions'],
            sl_learning_rate=checkpoint['sl_learning_rate'],
            train_every=checkpoint['train_every'],
            evaluate_with=checkpoint['evaluate_with'],
            device=checkpoint['device'],
            q_mlp_layers=checkpoint['rl_agent']['q_estimator']['mlp_layers'],
            state_shape=checkpoint['rl_agent']['q_estimator']['state_shape'],
            hidden_layers_sizes=[],
        )

        agent.policy_network = AveragePolicyNetwork.from_checkpoint(checkpoint['policy_network'])
        agent._reservoir_buffer = ReservoirBuffer.from_checkpoint(checkpoint['reservoir_buffer'])
        agent._mode = checkpoint['mode']
        agent.total_t = checkpoint['total_t']
        agent.train_t = checkpoint['train_t']
        agent.policy_network.to(agent.device)
        agent.policy_network.eval()
        agent.policy_network_optimizer = torch.optim.Adam(agent.policy_network.parameters(), lr=agent._sl_learning_rate)
        agent.policy_network_optimizer.load_state_dict(checkpoint['policy_network_optimizer'])
        agent._rl_agent.from_checkpoint(checkpoint['rl_agent'])
        agent._rl_agent.set_device(agent.device)
        return agent

    def save_checkpoint(self, path, filename='checkpoint_nfsp.pt'):
        ''' Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
        '''
        torch.save(self.checkpoint_attributes(), os.path.join(path, filename))


class AveragePolicyNetwork(nn.Module):
    """
    给定状态(平均策略)的行动概率的历史近似值。前向传递返回动作的对数概率
    """
    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        """
        初始化策略网络, 它只是一堆ReLU层, 最后一层没有激活
        用Xavier初始化(sonnet.nets.MLP和tensorflow默认值)
        输出动作的数量及每个样本的状态张量形状，每个mlp层的输出大小
        """
        super(AveragePolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        if not self.state_shape:
            self.state_shape = [[664], [664]]
        # 设置MLP层和ReLU激活函数
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        mlp = [nn.Flatten()]
        mlp.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            mlp.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims) - 2:
                mlp.append(nn.ReLU())

        # 构建网络顺序层
        self.mlp = nn.Sequential(*mlp)

    def forward(self, s):
        """
        记录状态中每个动作的动作概率
        """
        log_its = self.mlp(s)
        log_action_prob = F.log_softmax(log_its, dim=-1)
        return log_action_prob

class ReservoirBuffer(object):
    """
    储层缓冲器
    允许对数据流进行统一采样, 该类支持任意元素的存储，例如观察张量、整数动作等
    更多细节: https://en.wikipedia.org/wiki/Reservoir_sampling
    """
    def __init__(self, reservoir_buffer_capacity):
        """
        初始化存储缓冲区参数
        """
        # 存储缓冲器容量大小
        self._data = []
        self._add_calls = 0
        self._reservoir_buffer_capacity = reservoir_buffer_capacity

    def add(self, element):
        """
        添加数据至存储缓冲区
        """
        # 当数据容量小于缓冲器容量大小时，则添加
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """
        返回从存储缓冲区采样的样本数据，并进行迭代
        """
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        """
        清除缓存数据
        """
        self._data = []
        self._add_calls = 0

    def checkpoint_attributes(self):
        return {
            'data': self._data,
            'add_calls': self._add_calls,
            'reservoir_buffer_capacity': self._reservoir_buffer_capacity,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        reservoir_buffer = cls(checkpoint['reservoir_buffer_capacity'])
        reservoir_buffer._data = checkpoint['data']
        reservoir_buffer._add_calls = checkpoint['add_calls']
        return reservoir_buffer

    def __len__(self):
        """
        数据大小
        """
        return len(self._data)

    def __iter__(self):
        """
        迭代数据
        """
        return iter(self._data)