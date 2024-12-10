# -*- coding: utf-8 -*-

from reinforce.share.base_card import BaseCard


class MonsterCards(BaseCard):
    """ 对象 """
    # 方块
    FK_3 = 103, 1, 3
    FK_5 = 105, 1, 5
    FK_8 = 108, 1, 8
    FK_10 = 110, 1, 10
    FK_J = 111, 1, 11
    FK_Q = 112, 1, 12
    FK_K = 113, 1, 13
    # 梅花
    MH_3 = 203, 2, 3
    MH_5 = 205, 2, 5
    MH_8 = 208, 2, 8
    MH_10 = 210, 2, 10
    MH_J = 211, 2, 11
    MH_Q = 212, 2, 12
    MH_K = 213, 2, 13
    # 红心
    HX_3 = 303, 3, 3
    HX_5 = 305, 3, 5
    HX_8 = 308, 3, 8
    HX_10 = 310, 3, 10
    HX_J = 311, 3, 11
    HX_Q = 312, 3, 12
    HX_K = 313, 3, 13
    # 黑桃
    HT_3 = 403, 4, 3
    HT_5 = 405, 4, 5
    HT_8 = 408, 4, 8
    HT_10 = 410, 4, 10
    HT_J = 411, 4, 11
    HT_Q = 412, 4, 12
    HT_K = 413, 4, 13
    # 万能牌
    UNIVERSAL_CARD = 520, 5, 20
    # 捡牌
    PICK_CARD = 621, 6, 21

    @classmethod
    def all_cards(cls):
        """
        Get all cards
        """
        all_member = list(cls._member_map_.values())
        all_member.pop()  # remove universal card
        all_member.pop()  # remove pick card
        return all_member

    @classmethod
    def extra_cards(cls):
        """
        Get extra cards
        """
        return [MonsterCards.UNIVERSAL_CARD for _ in range(4)]