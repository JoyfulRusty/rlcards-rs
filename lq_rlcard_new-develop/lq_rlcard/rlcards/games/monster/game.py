# -*- coding: utf-8 -*-

import numpy as np


from rlcards.games.monster.judge import MonsterJudge as Judge
from rlcards.games.monster.round import MonsterRound as Round
from rlcards.games.monster.dealer import MonsterDealer as Dealer
from rlcards.games.monster.player import MonsterPlayer as Player

from rlcards.const.monster.const import PLAYER_NUMS, ALL_GOLDS, BUST_FLAGS, ROUND_CARDS


class MonsterGame:
    """
    游戏流程(打妖怪)
    """
    def __init__(self, allow_step_back=False):
        """
        游戏流程相关属性参数
        """
        self.num_players = PLAYER_NUMS
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()

    def init_game(self):
        """
        初始化游戏环境参数
        """
        self.winner_id = None  # 赢家ID
        self.landlord_id = None  # 庄家ID
        self.curr_state = None  # 当前状态
        self.pid_to_golds = []  # 游戏结束玩家的金币数量
        self.golds = ALL_GOLDS[:]  # 玩家初始金币数量
        self.bust = BUST_FLAGS # 添加破产玩家默认值
        self.round_cards = ROUND_CARDS[:]  # 游戏中所有卡牌(32张)

        self.judge = Judge(self.np_random)
        self.dealer = Dealer(self.np_random)
        self.round = Round(self.np_random, self.round_cards, self.dealer)
        self.players = [Player(num, self.np_random) for num in range(self.num_players)]

        # 持有方块J玩家
        self.landlord_id = self.dealer.deal_cards(self.players)
        self.round.curr_player_id = self.landlord_id

        curr_player_id = self.round.curr_player_id
        state = self.get_state(curr_player_id)
        self.curr_state = state

        return self.curr_state, curr_player_id

    def step(self, action):
        """
        迭代更新玩家动作选取和判断
        """
        curr_player = self.players[self.round.curr_player_id]

        # TODO: 删除打出卡牌(用于记录游戏是否结束)
        if isinstance(action, int) and action in self.round_cards:
            self.round_cards.remove(action)
            self.dealer.play_card(curr_player, action)
            # 记录手牌打完
            if not curr_player.curr_hand_cards:
                curr_player.is_all = True

        # TODO: 判断游戏是否结束(手牌出完则结束，包括破产玩家手牌)
        if self.judge.judge_name(self.round_cards, self.players):
            winner_id, account_golds = self.calc_winner_id(self.golds)
            self.winner_id = winner_id
            self.pid_to_golds = account_golds

        # TODO: 计算玩家其他操作
        else:
            # 判断操作流程，及破产后计算金币
            pay_after_golds, next_player_id = self.round.proceed_round(
                curr_player,
                self.players,
                action,
                self.golds,
                self.bust,
                self.curr_state,
            )

            self.round.curr_player_id = next_player_id
            state = self.get_state(next_player_id)
            self.curr_state = state
            return state, next_player_id

        state = self.get_state(curr_player.player_id)
        self.curr_state = state
        return state, curr_player.player_id

    def get_state(self, curr_player_id):
        """
        获取玩家状态
        """
        curr_player = self.players[curr_player_id]
        other_hand_cards = self.get_others_hand_cards(curr_player)
        hand_card_nums = [len(self.players[i].curr_hand_cards) for i in range(self.num_players)]

        state = self.round.get_state(
            curr_player,
            self.dealer,
            other_hand_cards,
            hand_card_nums,
            self.bust,
            self.round_cards,
        )

        return state

    def get_legal_actions(self):
        """
		获取玩家合法动作
		"""
        return self.curr_state['actions']

    @staticmethod
    def calc_winner_id(golds):
        """
        计算获胜玩家ID
        """
        p_id = [_ for _ in range(len(golds))]
        new_golds = dict(zip(p_id, golds))

        pid_to_golds = sorted(new_golds.items(), key=lambda x: x[1], reverse=True)

        winner_id = pid_to_golds[0][0]

        return winner_id, pid_to_golds

    def get_others_hand_cards(self, curr_p):
        """
        获取其他玩家(三位)当前手牌
        """
        # 其他三位玩家索引
        others_hand_cards = {'down': [], 'right': [], 'up': [], 'left': []}
        for player in self.players:
            if player.player_id == curr_p.player_id:
                continue
            if player.player_id == 0:
                others_hand_cards['down'].extend(player.curr_hand_cards)
            elif player.player_id == 1:
                others_hand_cards['right'].extend(player.curr_hand_cards)
            elif player.player_id == 2:
                others_hand_cards['up'].extend(player.curr_hand_cards)
            elif player.player_id == 3:
                others_hand_cards['left'].extend(player.curr_hand_cards)
        return others_hand_cards

    @staticmethod
    def get_num_actions():
        """
        返回抽象动作数量
        """
        return 30

    def get_player_id(self):
        """
        获取当前玩家ID
        """
        return self.round.curr_player_id

    def get_num_players(self):
        """
        获取玩家数量
        """
        return self.num_players

    def is_over(self):
        """
        判断游戏是否结束
        """
        # 判断是否有赢家
        if not self.round_cards:
            return True
        if self.winner_id is not None:
            return True
        return False