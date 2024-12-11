RECORD_LOG = False
FIRST_PLAY = 407
HAND_NUM = 13  # 手牌数


class MovesGener(object):
    """
    This is for generating the possible combinations
    """

    def __init__(self, info_set):
        self.info_set = info_set
        self.player_position = info_set.player_position
        self.cards_list = info_set.player_hand_cards
        self.light_cards = info_set.light_cards[info_set.player_position] if info_set.light_cards else []  # 当前玩家亮的牌
        self.played_cards = info_set.played_cards[info_set.player_position] if info_set.played_cards else []  # 当前玩家打过的牌
        self.suit_cards = {}  # 以花色区分手牌
        self.suit_cards_played = {}  # 打过的牌以花色区分

    def get_suit_cards(self, suit):
        """ 获取以花色区分的牌 """
        if not self.suit_cards:
            for c in self.cards_list:
                self.suit_cards.setdefault(c // 100, []).append(c)
        return self.suit_cards.get(suit) or []

    def get_suit_cards_by_played(self, suit):
        """ 获取以花色区分的牌 """
        if not self.suit_cards_played:
            for c in self.played_cards:
                self.suit_cards_played.setdefault(c // 100, []).append(c)
        return self.suit_cards_played.get(suit) or []

    def limit_cp_to_light_pai(self, must_suit=None):
        """ 亮牌后，若有亮牌花色的其他牌，第一次不得出亮的牌 """
        if not self.light_cards:
            return self.cards_list
        if not set(self.cards_list).intersection(self.light_cards):
            return self.cards_list
        RECORD_LOG and print(self.player_position, "limit_cp_to_light_pai start", "跟牌花色：", must_suit)
        can_cards_list = self.cards_list[:]
        suit_list = [c // 100 for c in self.light_cards]
        if must_suit:
            if must_suit not in suit_list:
                return can_cards_list
            suit_idx = suit_list.index(must_suit)
            suit_cards = self.get_suit_cards(must_suit)
            # 亮的牌的 花色大于1且未打过该花色的牌
            if len(suit_cards) > 1 and not self.get_suit_cards_by_played(must_suit):
                can_cards_list.remove(self.light_cards[suit_idx])
                RECORD_LOG and print("减去亮牌限制出牌1：", self.cards_list, "排除", self.light_cards[suit_idx], "打过的牌：",
                                     self.played_cards)
        else:
            for i, suit in enumerate(suit_list):
                suit_cards = self.get_suit_cards(suit)
                # 亮的牌的 花色大于1且未打过该花色的牌
                if len(suit_cards) > 1 and not self.get_suit_cards_by_played(suit):
                    can_cards_list.remove(self.light_cards[i])
                    RECORD_LOG and print("减去亮牌限制出牌2：", self.cards_list, "排除", self.light_cards[i], "打过的牌：",
                                         self.played_cards)
        return can_cards_list

    def gen_can_play_cards(self, must_suit=None) -> list:
        """
        找出所有能出的cards
        must_suit: 表示优先出的花色
        return: [[card1], ...]
        """
        # 首出
        if len(self.cards_list) == HAND_NUM:
            if FIRST_PLAY in self.cards_list:
                return [[FIRST_PLAY]]
        if not must_suit:
            # 第一家出牌，判断玩家有没有出过亮牌的同花色牌
            can_cards_list = self.limit_cp_to_light_pai()
            return [[c] for c in can_cards_list]

        # 跟牌，若未出过亮过的牌且该花色的牌数>1则亮的牌不能打
        can_cards_list = self.limit_cp_to_light_pai(must_suit)
        can_actions = []  # 同花色的牌
        for c in can_cards_list:
            if self.same_suit(c, must_suit):
                can_actions.append([c])
        return can_actions or [[c] for c in can_cards_list]  # 无同花色时任意出

    @staticmethod
    def same_suit(c, suit):
        return c // 100 == suit

    # generate all possible moves from given cards
    def gen_moves(self):
        moves = []
        moves.extend(self.gen_can_play_cards())  # 主动出
        return moves
