# -*- coding: utf-8 -*-

import random
import numpy as np

from reinforce.games.monster.judge import MonsterJudge as Judge
from reinforce.games.monster.round import MonsterRound as Round
from reinforce.games.monster.dealer import MonsterDealer as Dealer
from reinforce.games.monster.player import MonsterPlayer as Player

from reinforce.const.monster.const import PLAYER_NUMS, BUST_FLAGS, ROUND_CARDS


class MonsterGame:
    """
    游戏流程(打妖怪)
    """

    def __init__(self, allow_step_back = False):
        """
        游戏流程相关属性参数
        """
        self.bust = {}
        self.init_golds = {}
        self.round_cards = []
        self.judge = None
        self.round = None
        self.dealer = None
        self.players = None
        self.winner_id = None
        self.curr_state = None
        self.landlord_id = None
        self.over_round_golds = []
        self.num_players = PLAYER_NUMS
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()

    def init_game(self):
        """
        初始化游戏环境参数
        """
        self.winner_id = None
        self.curr_state = None
        self.landlord_id = None

        self.judge = Judge(self.np_random)
        self.dealer = Dealer(self.np_random)
        self.round = Round(self.np_random, self.round_cards, self.dealer)
        self.players = [Player(num, self.np_random) for num in range(self.num_players)]

        self.bust = BUST_FLAGS
        self.over_round_golds = []
        self.round_cards = ROUND_CARDS[:]
        self.init_golds = self.init_random_golds()
        self.landlord_id = self.dealer.deal_cards(self.players)
        self.round.curr_player_id = self.landlord_id
        curr_player_id = self.round.curr_player_id
        self.curr_state = self.get_state(curr_player_id)

        return self.curr_state, curr_player_id

    def init_random_golds(self):
        """
        初始化随机金币[800.0, 1600.0]
        """
        tmp_golds = {}
        for p_id in range(len(self.players)):
            tmp_golds[p_id] = round(random.uniform(800.0, 1600.0), 4)
        return tmp_golds

    def step(self, action):
        """
        todo: 更新迭代流程
        """
        curr_p = self.players[self.round.curr_player_id]

        # 移除标记已经打出卡牌(用于记录游戏是否结束)
        if isinstance(action, int) and action in self.round_cards:
            self.round_cards.remove(action)
            if action in curr_p.curr_hand_cards:
                self.dealer.play_card(curr_p, action)
            if not curr_p.curr_hand_cards:
                curr_p.is_all = True

        # todo: game over
        # 判断游戏是否结束(手牌出完则结束，包括破产玩家手牌)
        if self.judge.judge_name(self.round_cards, self.players):
            winner_id, final_golds = self.calc_winner_id(self.init_golds)
            self.winner_id = winner_id
            self.over_round_golds = final_golds

        # todo: other operations
        # 判断其他流程及动作
        else:
            next_player_id = self.round.proceed_round(
                curr_p,
                self.players,
                action,
                self.init_golds,
                self.bust,
                self.curr_state
            )
            # 迭代下一位玩家状态数据
            self.round.curr_player_id = next_player_id
            self.curr_state = self.get_state(next_player_id)
            return self.curr_state, next_player_id

        # todo: game over
        self.curr_state = self.get_state(curr_p.player_id)
        return self.curr_state, curr_p.player_id

    def get_state(self, curr_id):
        """
        获取玩家状态
        """
        curr_p = self.players[curr_id]
        other_hand_cards = self.get_others_hand_cards(curr_p)
        hand_card_nums = [len(self.players[i].curr_hand_cards) for i in range(self.num_players)]
        return self.round.get_state(
            curr_p,
            self.dealer,
            other_hand_cards,
            hand_card_nums,
            self.bust,
            self.round_cards,
        )

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
        over_round_golds = sorted(golds.items(), key=lambda x: x[1], reverse=True)
        return over_round_golds[0][0], over_round_golds

    def get_others_hand_cards(self, curr_p):
        """
        获取其他玩家(三位)当前手牌
        """
        others_hand_cards = {'down': [], 'right': [], 'up': [], 'left': []}
        for other_p in self.players:
            if other_p is curr_p:
                continue
            position = {
                0: "down",
                1: "right",
                2: "up",
                3: "left"
            }.get(other_p.player_id, 0)
            others_hand_cards[position].extend(other_p.curr_hand_cards)
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
        if not self.round_cards:
            return True
        return True if self.winner_id is not None else False