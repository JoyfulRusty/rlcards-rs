# -*- coding: utf-8 -*-

from reinforce.share.base_poker import BasePoker
from reinforce.games.monster_v2.card import MonsterCards


class MonsterPoker(BasePoker):
    """
    Monster Poker
    """

    CARDS_ENUM = MonsterCards

    def __init__(self):
        """
        Initialize Base Poker
        """
        super().__init__()

    def deal_cards(self, player_nums: int = 0, card_nums: int = 0, extra_nums: int = 0):
        """
        Deal cards to players
        params: player_count
        params: card_count
        params: extra_count
        """
        # 每人固定手牌数量
        all_cards = super().deal_cards(player_nums, card_nums)
        # 每人额外牌数量
        for _ in range(extra_nums):
            for i, c in enumerate(self.CARDS_ENUM.extra_cards()):
                all_cards[i].append(c)
        return all_cards