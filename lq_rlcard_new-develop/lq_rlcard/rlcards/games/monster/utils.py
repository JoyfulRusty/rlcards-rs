# -*- coding: utf-8 -*-

import numpy as np

from rlcards.const.monster import const
from rlcards.games.monster.card import Card


def new_make_cards():
    """
    卡牌编码字典
    """
    card_encoding_dict = {}
    num = 0
    for card in const.ALL_CARDS:
        card_encoding_dict[card] = num
        num += 1

    card_encoding_dict[const.MAGIC] = num
    card_encoding_dict[const.ActionType.PICK_CARDS] = num + 1

    return card_encoding_dict

# 构建动作编码字典
new_card_encoding_dict = new_make_cards()
card_decoding_dict = {new_card_encoding_dict[key]: key for key in new_card_encoding_dict.keys()}

def init_28_deck():
    """
    初始化卡牌
    """
    deck = []
    index_num = 0
    for card_value in const.ALL_CARDS:
        card_type = const.CARD_TYPES[card_value // 100]
        card = Card(card_type, card_value)
        card.set_index_num(index_num)
        index_num = index_num + 1
        deck.append(card)

    return deck

def magic_init_4_deck():
    """
    初始化万能牌
    """
    magic_deck = []
    index_num = 29
    card_type = const.CARD_TYPES[const.MAGIC // 100]
    card = Card(card_type, const.MAGIC)
    card.set_index_num(index_num)
    magic_deck.append(card)
    magic_deck = magic_deck * 4

    return magic_deck

def reorganize(trajectories, payoffs):
    """
    重新组织轨迹，使其对RL的学习效果更优
    """
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 2, 2):
            if i == len(trajectories[player] - 3):
                reward = payoffs[player]
                done = True
            else:
                reward, done = 0, False

            transition = trajectories[player][i: i+3].copy()
            transition.insert(2, reward)
            transition.append(done)
            new_trajectories[player].append(transition)

    return new_trajectories


def remove_illegal(actions_prob, legal_actions):
    """
    去掉非法行为，并对概率向量进行归一化
    """
    prob_s = np.zeros(actions_prob.shape[0])
    prob_s[legal_actions] = actions_prob[legal_actions]
    if np.sum(prob_s) == 0:
        prob_s[legal_actions] = 1 / len(legal_actions)
    else:
        prob_s /= sum(prob_s)

    return prob_s

def tournament(env, num):
    """
    评估环境
    """
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0

    while counter < num:
        _, _payoffs = env.run(is_training=False)
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
        payoffs[i] /= counter

    return payoffs

def compare2cards(card_1, card_2):
    """
    比较两张卡牌的大小
    """
    key = []
    for card in [card_1, card_2]:
        key.append(const.CARD_RANK.index(card % 100))
    if key[0] > key[1]:
        return 1
    if key[0] < key[1]:
        return -1
    return 0

def cards2str(cards):
    """
    获取卡牌对应的字符串表示
    """
    response = []
    for card in cards:
        response.append(card.card_value)
    return response