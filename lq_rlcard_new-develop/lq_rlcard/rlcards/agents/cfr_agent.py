# -*- coding: utf-8 -*-

import collections

import os
import pickle

from rlcards.utils.utils import *

class CFRAgent:
    """
    实施CFR(机会抽样)算法
    """
    def __init__(self, env, model_path='/cfr_model'):
        """
        初始化agent
        """
        self.env = env
        self.use_raw = False
        self.model_path = model_path

        # 策略和平均策略
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # 反事实遗憾值
        self.regrets = collections.defaultdict(np.array)
        self.interaction = 0

    def train(self):
        """
        CFR迭代
        """
        self.interaction = 1
        # 遍历树，用于计算每个玩家的反事实遗憾，将其记录在遍历中
        for player_id in range(self.env.num_players):
            self.env.reset()
            prob = np.ones(self.env.num_players)
            self.traverse_tree(prob, player_id)

        # 更新策略: 对缓存中的遗憾之进行更新
        self.update_policy()

    def traverse_tree(self, prob, player_id):
        """
        遍历游戏树，更新遗憾
        计算当前节点的到达概率，更新玩家值
        """
        # 判断游戏是否结束
        if self.env.is_over():
            # 获取对局奖励
            return self.env.get_payoffs()

        action = 0
        # 当前玩家ID
        current_player = self.env.get_player_id()
        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)

        # 状态信息数据与合法动作
        obs, legal_actions = self.get_states(current_player)
        # 计算动作概率
        action_prob = self.action_prob(obs, legal_actions, self.policy)

        for action in legal_actions:
            action_prob = action_prob[action]
            new_prob = prob.copy()
            new_prob[current_player] += action_prob

            # 更新下一个动作，反转树，更新遗憾值
            self.env.step(action)
            utility = self.traverse_tree(new_prob, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # 如果为缓存中玩家，将记录策略和遗憾值
        player_prob = prob[current_player]
        counterfactual_prob = (
            np.prod(prob[:current_player]),
            np.prod(prob[current_player + 1:])
        )

        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)

        if obs not in self.average_policy:
            action_prob = action_prob[action]
            regret = counterfactual_prob * (action_utilities[action][current_player] - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.interaction * player_prob * action_prob

        return state_utility

    def update_policy(self):
        """
        对缓存的遗憾值进行更新
        """
        for obs in self.regrets:
            self.policy[obs] = self.regrets_math(obs)

    def regrets_math(self, obs):
        """
        对遗憾值进行匹配(匹配遗憾值)
        """
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        # 动作概率分布
        action_prob = np.zeros(self.env.num_actions)

        # 有效反事实遗憾值，优先选择
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_prob[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_prob[action] = 1.0 / self.env.num_actions

        return action_prob

    def action_prob(self, obs, legal_actions, policy):
        """
        获取当前状态的动作概率
        """
        if obs not in policy.keys():
            action_prob = np.array([1.0 / self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_prob
        else:
            action_prob = policy[obs]
        action_prob = remove_illegal(action_prob, legal_actions)

        return action_prob

    def eval_step(self, state):
        """
        给定状态，根据平均策略预测下一个操作动作
        """
        prob = self.action_prob(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(prob), p=prob)
        info = dict()
        info['prob'] = {
            state['raw_legal_action'][i]: float(prob[list(state['legal_actions'].keys())[i]])
            for i in range(len(state['legal_actions']))
        }

        return action, info

    def get_states(self, player_id):
        """
        获取玩家状态
        """
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['actions'].keys())

    def save(self):
        """
        模型保存
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'), 'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'), 'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'), 'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        interaction_file = open(os.path.join(self.model_path, 'interaction_file.pkl'), 'wb')
        pickle.dump(self.interaction, interaction_file)
        interaction_file.close()

    def load(self):
        """
        加载模型
        """
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'), 'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'), 'rb')
        self.average_policy = pickle.load(policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'), 'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        interaction_file = open(os.path.join(self.model_path, 'interaction_file.pkl'), 'rb')
        self.interaction = pickle.load(interaction_file)
        interaction_file.close()