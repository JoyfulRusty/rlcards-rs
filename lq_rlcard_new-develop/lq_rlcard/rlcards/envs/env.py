# -*- coding: utf-8 -*-

from rlcards.utils import seeding


class Env:
    """
    卡牌游戏基础环境
    """
    def __init__(self, config, game):
        """
        初始化基础环境属性和参数
        """
        self.game = game  # 游戏环境(ddz, monster, mahjong...)
        self.agents = []
        self.time_step = 0
        self.np_random = 0
        self.game_np_random = 0
        self.action_recorder = []
        self.seed(config['seed'])
        self.num_players = self.game.get_num_players()
        self.num_actions = self.game.get_num_actions()
        self.allow_step_back = self.game.allow_step_back = config['allow_step_back']

    def reset(self):
        """
        初始化游戏环境
        """
        state, player_id = self.game.init_game()
        self.action_recorder = []

        return self.extract_state(state), player_id

    def step(self, action, raw_action=False):
        """
        更新解码动作和下一个玩家状态
        """
        # 根据动作索引，解析对应动作
        if not raw_action:
            action = self._decode_action(action)

        # 时间步长
        self.time_step += 1

        # 出牌记录
        self.action_recorder.append((self.get_player_id(), action))

        # 更新下一位玩家状态
        next_state, player_id = self.game.step(action)

        return self.extract_state(next_state), player_id

    def set_agents(self, agents):
        """
        模型代理
        """
        self.agents = agents

    def run(self, is_training=False):
        """
        TODO: 执行游戏环境，收集玩家对局数据，用于训练
        """
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()
        trajectories[player_id].append(state)

        # TODO: 训练流程
        while not self.is_over():
            if not is_training:
                # 评估模式
                action, _ = self.agents[player_id].eval_step(state)
            else:
                # 训练模式(预测输出动作)
                action = self.agents[player_id].step(state)

            # 向轨迹中添加出牌动作
            trajectories[player_id].append(action)

            # 更新下一位玩家状态
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)

            state = next_state
            player_id = next_player_id

            # 判断游戏是否符合结束条件
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # 游戏结束，添加所有玩家state数据
        for played_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # 计算对局奖励收益
        payoffs = self.get_payoffs()

        return trajectories, payoffs, self.num_players, player_id

    def step_back(self):
        """
        todo: 回溯更新
        """
        if not self.allow_step_back:
            raise Exception('Step back is off. To use step_back, please set allow_step_back=True in rlcard.make')

        if not self.game.step_back():
            return False

        player_id = self.get_player_id()
        state = self.get_state(player_id)

        return state, player_id

    def is_over(self):
        """
        游戏是否结束
        """
        return self.game.is_over()

    def get_player_id(self):
        """
        当前玩家ID
        """
        return self.game.get_player_id()

    def get_state(self, player_id):
        """
        状态数据
        """
        return self.extract_state(self.game.get_state(player_id))

    def seed(self, seed=None):
        """
        创建随机数
        """
        self.np_random, seed = seeding.np_random(seed)
        self.game_np_random = self.np_random
        return seed

    def get_payoffs(self):
        """
        对局收益
        """
        raise NotImplementedError

    def extract_state(self, state):
        """
        抽取状态数据
        """
        raise NotImplementedError

    def _decode_action(self, action_id):
        """
        解码动作
        """
        raise NotImplementedError

    def _get_legal_actions(self):
        """
       合法动作
        """
        raise NotImplementedError