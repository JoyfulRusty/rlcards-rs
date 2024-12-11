# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# todo: 伯努利多臂老虎机
'''
1.期望奖励Q(a) = Er~R(·|a)[R], 最优期望奖励Q* = maxQ(a)
2.懊悔: 定义为动作a期望奖励与最优拉杆的期望奖励差 -> R(a) = Q* - Q(a)
3.更新计数器: N(at) = N(at) + 1, 期望奖励估值Q^(ai)
4.更新期望奖励估值: Q^(at) = Q^(at) + (1/N(at) * [rt - Q^(at)])
5.增量式期望更新： Qk = Qk-1 + (1/k * [rk - Qk-1])
6.£-贪心策略： at = [1.argmaxQ^(a), 采样概率为: 1-£, 2.从A中随机选择，采样概率为: £]
'''


class BernoulliBandit:
    """
    伯努利多臂老虎机,输入K表示拉杆个数
    """

    def __init__(self, k):
        """
        初始化参数
        """
        self.action_prob = np.random.uniform(size=k)  # 随机生成K个0~1的数，作为每根拉杆的获奖
        self.best_idx = np.argmax(self.action_prob)  # 获奖概率最大的拉杆
        self.best_prob = self.action_prob[self.best_idx]  # 最大的获奖概率
        self.K = k

    def step(self, k):
        """
        选择k号拉杆后，根据该老虎机的k号拉杆获得奖励的概率返回1(获奖) or 0(未获奖)
        """
        if np.random.rand() < self.action_prob[k]:
            return 1
        else:
            return 0


# 测试
np.random.seed(1)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))


class Solver:
    """
    多臂老虎机算法基本框架
    """

    def __init__(self, bandit):
        """
        初始化参数
        """
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累计懊悔
        self.actions = []  # 维护一个列表，记录每一步的动作
        self.regrets = []  # 维护一个列表，记录每一步的懊悔累计

    def update_regret(self, k):
        """
        计算累计懊悔并保存，k为本次动作选取的栏杆编号
        """
        self.regret += self.bandit.best_prob - self.bandit.action_prob[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """
        返回当前动作选择那一根拉杆，由每个具体的策略实现
        """
        raise NotImplementedError

    def run(self, num_steps):
        """
        运行一定的次数，num_steps为总运行次数
        """
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """
    Epsilon贪婪算法,继承Solver类
    """

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        """
        初始化参数
        """
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化拉动所有拉杆的期望奖励估值

    def run_one_step(self):
        """
        运行更新
        """
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆

        r = self.bandit.step(k)  # 更新本次动作奖励
        # Q(at) = Q(at) + 1/N(at) * (rt - Q`(at))
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


def plot_results(solvers, solver_names):
    """
    生成累计懊悔随时间变化的图像，输入solvers是一个列表，列表中的每个元素是一种特定的策略
    solver_names也是一个列表，存储每个策略的名称
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


# 测试结果
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


# np.random.seed(0)
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list:
# 	solver.run(5000)
#
# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


class DecayingEpsilonGreedy(Solver):
    """
    Epsilon值随时间衰减epsilon-贪婪算法，继承Solver类
    """

    def __init__(self, bandit, init_prob=1.0):
        """
        初始化参数
        """
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        """
        运行更新
        """
        self.total_count += 1
        # epsilon值随时间衰减
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        # Q(at) = Q(at) + 1/N(at) * (rt - Q`(at))
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


# 测试结果
np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


class UCB(Solver):
    """
    UCB算法，继承Solver类
    """

    def __init__(self, bandit, cef_prob, init_prob=1.0):
        """
        初始化参数
        """
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.cef_prob = cef_prob

    def run_one_step(self):
        """
        运行更新
        """
        self.total_count += 1
        ucb = self.estimates + self.cef_prob * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))  # 计算上置信界
        )

        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


# 测试结果
np.random.seed(1)
cef_prob = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, cef_prob)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])


class ThompsonSampling(Solver):
    """
    汤姆森采样算法，继承Solver类
    """

    def __init__(self, bandit):
        """
        初始化参数
        """
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表，表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表，表示每根拉杆奖励为0的次数

    def run_one_step(self):
        """
        运行更新
        """
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)
        self._a[k] += r  # 更新Beta分布的第一参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数

        return k


# 测试结果
np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])
