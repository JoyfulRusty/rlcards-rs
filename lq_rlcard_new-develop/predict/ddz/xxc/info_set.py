# -*- coding: utf-8 -*-

class InfoSet:
    """
    The class to store the information set for the current player.
    """
    __slots__ = (
        "player_position", "player_hand_cards", "num_cards_left_dict",
        "three_landlord_cards", "card_play_action_seq", "other_hand_cards",
        "legal_actions", "last_move", "last_two_moves", "last_move_dict",
        "played_cards", "all_hand_cards", "last_pid", "bomb_num"
    )

    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand hands of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict = None
        # The three landlord cards. A list.
        self.three_landlord_cards = None
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the positions
        self.last_move_dict = None
        # The played hands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict.
        self.all_hand_cards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
        # The number of bombs played so far
        self.bomb_num = None

info_set = InfoSet("landlord")


class InfoSetAn(object):
    """
    4人斗地主观测
    """
    __slots__ = (
        "player_position", "player_hand_cards", "num_cards_left_dict", "card_play_action_seq", "other_hand_cards",
        "legal_actions", "last_move", "last_two_moves", "last_move_dict",
        "played_cards", "all_hand_cards", "bomb_num", "played_fk_3", "has_fk_3_di_zhu"
    )

    def __init__(self, player_position):
        # 1The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # 1The hand hands of the current player. A list.
        self.player_hand_cards = None  # todo: 地主牌
        # 1The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict = None
        # 1The historical moves. It is a list of list
        self.card_play_action_seq = None
        # 1The union of the hand cards of the other two players for the current player
        self.other_hand_cards = None  # todo: 地主牌
        # 1The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # 1The most recent valid move
        self.last_move = None
        # 1The most recent two moves
        self.last_two_moves = None
        # 1The last moves for all the positions
        self.last_move_dict = None
        # 1The played hands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict.
        self.all_hand_cards = None
        # 1The number of bombs played so far
        self.bomb_num = None
        self.played_fk_3 = None  # 方块三已出？
        self.has_fk_3_di_zhu = None  # 是持有方块3的地主？


info_set_an = InfoSetAn("landlord1")