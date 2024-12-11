from rlcards.games.pig.env.move_generator import MovesGener


def get_move_type(move):
    move_size = len(move)
    if move_size == 0:
        return {'type': 0}
    return {'type': 1, 'suit': move[0] // 100}


def get_legal_card_play_actions(infoset):
    """ 获取合法动作 """
    mg = MovesGener(infoset)

    rival_move = infoset.last_max_move
    rival_type = get_move_type(rival_move)
    rival_move_type = rival_type['type']
    rival_suit = rival_type.get('suit', None)
    moves = list()

    if rival_move_type == 0:
        moves = mg.gen_moves()
    elif rival_move_type == 1:
        moves = mg.gen_can_play_cards(rival_suit)

    for m in moves:
        m.sort()
    return moves


# -*- coding: utf-8 -*-
class InfoSet(object):
    """
    state info
    """
    __instance = None

    def __new__(cls, *args):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, player_position):
        self.player_position = player_position  # 玩家位置：landlord1, landlord2...
        self.light_cards = None  # 亮的牌1
        self.player_hand_cards = None  # 手牌1
        self.played_cards = None  # 每个玩家历史出牌1
        self.last_max_move = None  # 该轮中最大的move1
        self.legal_actions = None  # 合法动作1
        self.other_hand_cards = None  # 其它人的手牌1
        self.receive_score_cards = None  # 收到的分牌1
        self.remain_score_cards = None  # 剩余分牌1
        self.card_play_action_seq = None  # 所有玩家打牌序列list

    def update_state(self, info):
        # 1.每个玩家的亮牌
        # 2.手牌
        # 3.合法动作
        # 4.每个玩家出过的牌
        # 5.其它手牌
        # 6.该轮此前最大的move
        # 7.每名玩家收的分牌
        # 8.剩余分牌
        self.player_position = info.get("player_position") or ""
        self.light_cards = info.get("light_cards") or {}
        hand_cards = info.get("cards") or []
        hand_cards.sort()
        self.player_hand_cards = hand_cards
        self.played_cards = info.get("played_cards") or []
        self.last_max_move = info.get("last_max_move") or []
        self.legal_actions = get_legal_card_play_actions(self)
        self.other_hand_cards = info.get("other_hand") or []
        self.receive_score_cards = info.get("receive_score_cards") or []
        self.remain_score_cards = info.get("remain_score_cards") or []
        self.card_play_action_seq = info.get("card_play_action_seq") or []


info_set = InfoSet("landlord1")