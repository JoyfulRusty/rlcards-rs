from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

bombs = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]

class GameEnv(object):

    def __init__(self, players):
        # 游戏参数
        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.players = players

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.num_wins = {'landlord': 0,
                         'farmer': 0}

        self.num_scores = {'landlord': 0,
                           'farmer': 0}

        # 初始化玩家信息[landlord, landlord_down, landlord_up]
        # 相当于初始化三位玩家的属性参数
        self.info_sets = {
            'landlord': InfoSet('landlord'),
            'landlord_up': InfoSet('landlord_up'),
            'landlord_down': InfoSet('landlord_down')
        }

        self.bomb_num = 0
        self.last_pid = 'landlord'

    def card_play_init(self, card_play_data):
        """
        设置玩家卡牌
        """
        self.info_sets['landlord'].player_hand_cards = \
            card_play_data['landlord']
        self.info_sets['landlord_up'].player_hand_cards = \
            card_play_data['landlord_up']
        self.info_sets['landlord_down'].player_hand_cards = \
            card_play_data['landlord_down']
        self.three_landlord_cards = card_play_data['three_landlord_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        """
        判断游戏是否结束
        其中任何一位玩家手牌打完，则游戏结束
        """
        if len(self.info_sets['landlord'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_up'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_down'].player_hand_cards) == 0:

            # 计算是地主赢还是农民
            self.compute_player_utility()
            # 更新获胜者分数
            self.update_num_wins_scores()

            self.game_over = True

    def compute_player_utility(self):
        """
        计算地主赢还是农民
        """
        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            self.player_utility_dict = {
                'landlord': 2,
                'farmer': -1
            }
        else:
            self.player_utility_dict = {
                'landlord': -2,
                'farmer': 1
            }

    def update_num_wins_scores(self):
        """
        更新赢家的分数
        """
        for pos, utility in self.player_utility_dict.items():
            base_score = 2 if pos == 'landlord' else 1
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                self.num_scores[pos] += base_score * (2 ** self.bomb_num)
            else:
                self.num_scores[pos] -= base_score * (2 ** self.bomb_num)

    def get_winner(self):
        """
        获胜赢家
        """
        return self.winner

    def get_bomb_num(self):
        """
        获取炸弹数量
        """
        return self.bomb_num

    def step(self):
        action = self.players[self.acting_player_position].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if len(action) > 0:
            self.last_pid = self.acting_player_position

        if action in bombs:
            self.bomb_num += 1

        # 三位玩家出牌
        self.last_move_dict[
            self.acting_player_position] = action.copy()

        self.card_play_action_seq.append(action)
        # 更新出牌玩家手牌
        # 删除玩家打出的手牌
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_position] += action

        if self.acting_player_position == 'landlord' and \
                len(action) > 0 and \
                len(self.three_landlord_cards) > 0:
            for card in action:
                if len(self.three_landlord_cards) > 0:
                    if card in self.three_landlord_cards:
                        self.three_landlord_cards.remove(card)
                else:
                    break

        self.game_done()
        if not self.game_over:
            # 获取下一位出牌玩家
            self.get_acting_player_position()
            # 获取位置信息
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        # 玩家的出牌
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]

        return last_move

    def get_last_two_moves(self):
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_acting_player_position(self):
        """
        更新下一位玩家
        """
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'

        else:
            if self.acting_player_position == 'landlord':
                self.acting_player_position = 'landlord_down'

            elif self.acting_player_position == 'landlord_down':
                self.acting_player_position = 'landlord_up'

            else:
                self.acting_player_position = 'landlord'

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action != []:
            for card in action:
                self.info_sets[
                    self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        mg = MovesGener(
            self.info_sets[self.acting_player_position].player_hand_cards)

        action_sequence = self.card_play_action_seq

        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == md.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = ms.filter_type_13_4_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_14_4_22:
            all_moves = mg.gen_type_14_4_22()
            moves = ms.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [md.TYPE_0_PASS,
                                   md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def reset(self):
        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.info_sets = {
            'landlord': InfoSet('landlord'),
            'landlord_up': InfoSet('landlord_up'),
            'landlord_down': InfoSet('landlord_down')}

        self.bomb_num = 0
        self.last_pid = 'landlord'

    def get_infoset(self):
        # 上一位出牌玩家
        self.info_sets[self.acting_player_position].last_pid = self.last_pid
        # 玩家的合法动作
        self.info_sets[self.acting_player_position].legal_actions = self.get_legal_card_play_actions()

        self.info_sets[self.acting_player_position].bomb_num = self.bomb_num
        # 上一轮出牌
        self.info_sets[self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[self.acting_player_position].last_two_moves = self.get_last_two_moves()

        # 玩家的出牌记录
        self.info_sets[self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {
                pos: len(self.info_sets[pos].player_hand_cards)
                for pos in ['landlord', 'landlord_up', 'landlord_down']
            }

        # 所有玩家的卡牌
        self.info_sets[self.acting_player_position].other_hand_cards = []
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += self.info_sets[pos].player_hand_cards
        # 出牌
        self.info_sets[self.acting_player_position].played_cards = self.played_cards
        # 地主牌
        self.info_sets[self.acting_player_position].three_landlord_cards = self.three_landlord_cards
        # 桌牌
        self.info_sets[self.acting_player_position].card_play_action_seq = self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_hand_cards = {
            pos: self.info_sets[pos].player_hand_cards for pos in ['landlord', 'landlord_up', 'landlord_down']
        }

        return deepcopy(self.info_sets[self.acting_player_position])

class InfoSet(object):
    """
    游戏状态被描述为info_set包括当前情况下的所有信息、例如三个玩家的手牌、历史动作等
    """
    def __init__(self, player_position):
        # 玩家位置信息: landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # 玩家当前卡牌
        self.player_hand_cards = None
        # 每一位玩家的卡牌数量
        self.num_cards_left_dict = None
        # 地主牌
        self.three_landlord_cards = None
        # 历史打牌信息
        self.card_play_action_seq = None
        # 其他玩家的卡牌
        self.other_hand_cards = None
        # 当前出的合法动作
        self.legal_actions = None
        # 上一次出的一张牌
        self.last_move = None
        # 上一次出的两张牌
        self.last_two_moves = None
        # 已经打过牌的位置
        self.last_move_dict = None
        # 玩家卡牌
        self.played_cards = None
        # 所有玩家的卡牌
        self.all_hand_cards = None
        # 上一位玩家的有效操作，例如: pass
        self.last_pid = None
        # 炸弹数量
        self.bomb_num = None