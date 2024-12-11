from copy import deepcopy
from . import move_detector as md
from .move_generator import MovesGener
from .utils import SCORE_CARDS, BAN_CARD, ALL_ROLE, ALL_RED, MAN_GUAN_NUM
from .light_cards_policy import lp_policy

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

RECORD_LOG = False


class GameEnv(object):

    def __init__(self, players):

        self.card_play_action_seq = []
        self.turn_cards = []  # 自加，一轮中的牌
        self.first_play = None  # 自加
        self.remain_score_cards = []  # 自加，剩余分牌
        self.all_role = (
            'landlord1',
            'landlord2',
            'landlord3',
            'landlord4',
        )
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.players = players

        self.last_move_dict = {role: [] for role in ALL_ROLE}

        self.played_cards = {
            'landlord1': [],
            'landlord2': [],
            'landlord3': [],
            'landlord4': []
        }

        self.light_cards = {role: [] for role in ALL_ROLE}
        self.receive_score_cards = {role: [] for role in ALL_ROLE}

        self.last_move = []
        self.last_two_moves = []

        self.num_wins = {
            'landlord1': 0,
            'landlord2': 0,
            'landlord3': 0,
            'landlord4': 0,
        }

        # 评估用
        self.num_scores = {
            'landlord1': 0,
            'landlord2': 0,
            'landlord3': 0,
            'landlord4': 0,
        }

        self.info_sets = {
            'landlord1': InfoSet('landlord1'),
            'landlord2': InfoSet('landlord2'),
            'landlord3': InfoSet('landlord3'),
            'landlord4': InfoSet('landlord4'),
        }

        self.last_pid = 'landlord1'
        self.step_count = 0  # 新加

    def card_play_init(self, card_play_data):
        self.info_sets['landlord1'].player_hand_cards = \
            card_play_data['landlord1']
        self.info_sets['landlord2'].player_hand_cards = \
            card_play_data['landlord2']
        self.info_sets['landlord3'].player_hand_cards = \
            card_play_data['landlord3']
        self.info_sets['landlord4'].player_hand_cards = \
            card_play_data['landlord4']
        self.get_acting_player_position()
        self.score_cards_init()  # 初始化分牌
        self.start_light_cards()
        self.game_infoset = self.get_infoset()

    def score_cards_init(self):
        self.info_sets['landlord1'].curr_score = self.info_sets['landlord2'].curr_score = \
            self.info_sets['landlord3'].curr_score = self.info_sets['landlord4'].curr_score = 0

    def start_light_cards(self):
        """ 开始亮牌 """
        def get_yet_lp():
            """ 获取已经亮的牌 """
            lp_cards = []
            for _, lp in self.light_cards.items():
                lp_cards.extend(lp)
            return lp_cards

        for position in self.all_role:
            hand = self.info_sets[position].player_hand_cards
            cards = lp_policy(hand, get_yet_lp())
            self.light_cards[position] = cards
        RECORD_LOG and print("亮牌完成：", self.light_cards)

    def game_done(self):
        """
        1.四个玩家手牌完
        2.分牌已经被收完
        """
        if len(self.info_sets['landlord1'].player_hand_cards) == 0 and \
                len(self.info_sets['landlord2'].player_hand_cards) == 0 and \
                len(self.info_sets['landlord3'].player_hand_cards) == 0 and \
                len(self.info_sets['landlord4'].player_hand_cards) == 0:
            self.game_over = True

    def check_score_cards_over(self):
        """ 得在一轮结束收分处调用，不能和game_done放一起，因为每个step只是执行了一个action，而非收分 """
        self.remain_score_cards = self.cal_res_score()
        if self.remain_score_cards:
            RECORD_LOG and print("目前剩余分牌：", self.remain_score_cards)
            return False
        RECORD_LOG and print("分牌已经完了，游戏结束!!!")
        self.compute_player_utility()  # 收完最后一轮分再计算赢的次数
        self.update_num_wins_scores()
        self.game_over = True
        return True

    def get_reward(self):
        """ 新结算：采用真实算分再缩放 """
        diff_score = self.cal_res_score()
        if diff_score:
            raise ValueError("分牌不对，请检查！！", diff_score)
        reward_info = {}
        for role1 in self.all_role:
            other_score = 0
            for role2 in self.all_role:
                if role1 == role2:
                    continue
                other_score += self.info_sets[role2].curr_score
            if RECORD_LOG:
                print("结算：", role1, self.info_sets[role1].receive_score_cards, self.info_sets[role1].curr_score)
            role1_score = self.info_sets[role1].curr_score * 3 - other_score
            reward_info[role1] = role1_score
            self.num_scores[role1] += role1_score

        RECORD_LOG and print("玩家得分情况1：", reward_info, self.step_count)
        for role1 in self.all_role:
            reward_info[role1] = reward_info[role1] / 100 * (1.2 - self.step_count * 0.0033)
        RECORD_LOG and print("玩家得分情况2：", reward_info)
        return reward_info

    def cal_res_score(self):
        """
        计算剩余分牌
        """
        score_cards = self.info_sets["landlord1"].receive_score_cards
        all_score_cards = []
        for p, s in score_cards.items():
            all_score_cards.extend(s)
        diff_score = list(set(SCORE_CARDS.keys()).difference(all_score_cards))
        return diff_score

    def check_is_had_man_guan(self):
        """ 检测大满贯 """
        score_cards_len = len(self.receive_score_cards[self.acting_player_position])
        return score_cards_len == MAN_GUAN_NUM

    def check_is_had_quan_hong(self):
        """ 检测全红 """
        score_cards = self.receive_score_cards[self.acting_player_position]
        if not set(ALL_RED).difference(score_cards):
            return True
        return False

    def cal_man_guan_score(self):
        """ 大满贯计算分 """
        scores = 0
        for s_c in self.receive_score_cards[self.acting_player_position]:
            s = SCORE_CARDS.get(s_c)
            scores += abs(s)
        scores -= SCORE_CARDS.get(BAN_CARD)
        scores *= 2
        if RECORD_LOG:
            position = self.acting_player_position
            print(position, "大满贯：", self.receive_score_cards[position], scores)
        self.info_sets[self.acting_player_position].curr_score = scores  # 110

    def cal_quan_hong_score(self, has_ban=False):
        scores = 0
        for s_c in self.receive_score_cards[self.acting_player_position]:
            s = SCORE_CARDS.get(s_c)
            scores += abs(s)
        if has_ban:
            scores -= SCORE_CARDS.get(BAN_CARD)
            scores *= 2
        if RECORD_LOG:
            position = self.acting_player_position
            print(position, "全红：", self.receive_score_cards[position], scores)
        self.info_sets[self.acting_player_position].curr_score = scores

    def receive_score_cards_by_turn(self):
        """
        一轮结束找出分牌, 收入该轮牌最大的玩家手中
        并叠加分数
        """
        score_cards = []
        for c in self.turn_cards:
            if c in SCORE_CARDS:
                score_cards.append(c)
        if score_cards:
            self.receive_score_cards[self.acting_player_position].extend(score_cards)
            hand_has_ban = BAN_CARD in self.receive_score_cards[self.acting_player_position]
            if self.check_is_had_man_guan():
                self.cal_man_guan_score()
            elif self.check_is_had_quan_hong():
                self.cal_quan_hong_score(hand_has_ban)
            elif hand_has_ban:
                if len(self.receive_score_cards[self.acting_player_position]) == 1:
                    total_score = SCORE_CARDS.get(BAN_CARD)
                else:
                    # 有变压器则重算
                    total_score = 0
                    for c in self.receive_score_cards[self.acting_player_position]:
                        total_score += SCORE_CARDS.get(c, 0)
                    total_score -= SCORE_CARDS.get(BAN_CARD)
                    total_score *= 2
                self.info_sets[self.acting_player_position].curr_score = total_score
            else:
                add_score = 0
                for c in score_cards:
                    add_score += SCORE_CARDS.get(c, 0)
                self.info_sets[self.acting_player_position].curr_score += add_score
            if RECORD_LOG:
                print("此轮收分玩家：", self.acting_player_position, self.turn_cards,
                      "分牌: ", score_cards, self.receive_score_cards, "得分：",
                      self.info_sets[self.acting_player_position].curr_score)
            # 计算剩余分牌
            self.check_score_cards_over()

        self.turn_cards = []
        RECORD_LOG and print("一轮算分完成清空turn_cards")

    def compute_player_utility(self):
        self.player_utility_dict = self.get_reward()

    def update_num_wins_scores(self):
        self.winner = []
        for pos, utility in self.player_utility_dict.items():
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner.append(pos)

    def get_winner(self):
        return self.winner

    def get_bomb_num(self):
        return

    def step(self):
        action = self.players[self.acting_player_position].act(
            self.game_infoset)

        self.step_count += 1
        if len(action) > 0:
            self.last_pid = self.acting_player_position

        self.last_move_dict[
            self.acting_player_position] = action.copy()  # todo：每轮清空？？？

        self.card_play_action_seq.append(action)
        if len(self.turn_cards) < 4:
            self.turn_cards.append(action[0])
        else:
            # 记录一轮最新玩家出牌
            self.turn_cards = [action[0]]
        self.update_acting_player_hand_cards(action)
        self.played_cards[self.acting_player_position] += action
        if len(self.card_play_action_seq) > 52:
            raise ValueError("!!!牌数量错误")

        self.game_done()
        # RECORD_LOG and self.log_step_info(action)
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()
        else:
            if len(self.turn_cards) == 4:
                RECORD_LOG and print("next step 2")
                self.update_acting_player()  # 一轮完找出出牌最大玩家
                self.receive_score_cards_by_turn()  # 一轮收分

    def log_step_info(self, action):
        hand_cards = self.info_sets[self.acting_player_position].player_hand_cards[:] + action
        hand_cards.sort()
        print("------------------------------>")
        print("此轮最大的牌", self.info_sets[self.acting_player_position].last_max_move)
        print("curr_player: ", self.acting_player_position)
        print("player_hand_cards: ", hand_cards)
        print("legal_action: ", self.info_sets[self.acting_player_position].legal_actions)
        print("move: ", action)
        print("turn_cards: ", self.turn_cards)
        print("card_play_action_seq: ", self.card_play_action_seq)
        print("收分情况：", self.receive_score_cards)
        if self.game_over:
            print("player_utility_dict: ", self.player_utility_dict)
            print("score_cards: ", self.receive_score_cards)
            print("-----一局结束-------")
        print()

    def find_same_suit_card_by_turn(self):
        if not self.turn_cards:
            return []
        first_card = self.turn_cards[0]
        same_suit_cards = [first_card]
        suit = first_card // 100
        for c in self.turn_cards[1:]:
            if c // 100 == suit:
                same_suit_cards.append(c)
        return same_suit_cards

    def get_max_by_turn(self):
        """
        动态获取该轮中目前最大的牌
        如该轮只有2人出牌，则在这2个人出的牌中找
        """
        max_card = None
        same_suit_cards = self.find_same_suit_card_by_turn()
        for c in same_suit_cards:
            if not max_card:
                max_card = c
            elif c > max_card:
                max_card = c
        return [max_card] if max_card else []

    def get_last_move(self):
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
        """ 获取当前出牌玩家 """
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord1'
            self.first_play = 'landlord1'
        else:
            turn_len = len(self.turn_cards)
            if turn_len < 4:
                if self.acting_player_position == 'landlord1':
                    self.acting_player_position = 'landlord2'
                elif self.acting_player_position == 'landlord2':
                    self.acting_player_position = 'landlord3'
                elif self.acting_player_position == 'landlord3':
                    self.acting_player_position = 'landlord4'
                elif self.acting_player_position == 'landlord4':
                    self.acting_player_position = 'landlord1'
            else:
                RECORD_LOG and print("next step 1")
                self.update_acting_player()
                self.receive_score_cards_by_turn()  # 一轮收分

        return self.acting_player_position

    def update_acting_player(self):
        same_suit_cards = self.find_same_suit_card_by_turn()
        same_suit_cards.sort()
        index = self.turn_cards.index(same_suit_cards[-1])  # 找出一轮花色最大的

        max_index = int(int(self.first_play[-1]) + index - 1) % 4
        self.acting_player_position = self.all_role[max_index]
        self.first_play = self.acting_player_position
        # RECORD_LOG and print("一轮结束, first_player: ", self.first_play, self.turn_cards, "最大玩家index： ",
        #                      self.all_role[max_index])

    def update_acting_player_hand_cards(self, action: list):
        if action:
            curr_player_pos = self.acting_player_position
            for card in action:
                self.info_sets[curr_player_pos].player_hand_cards.remove(card)
            self.info_sets[curr_player_pos].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        """ 获取合法动作 """
        mg = MovesGener(self.info_sets[self.acting_player_position])

        turn_cards = self.turn_cards
        rival_move = []
        if len(turn_cards) != 0:
            rival_move = turn_cards[0:1]  # 须优先跟首家出一样花色

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_suit = rival_type.get('suit', None)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()
        elif rival_move_type == md.TYPE_1_SINGLE:
            moves = mg.gen_can_play_cards(rival_suit)

        for m in moves:
            m.sort()
        return moves

    def reset(self):
        self.card_play_action_seq = []
        self.turn_cards = []
        self.first_play = None
        self.remain_score_cards = []
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.last_move_dict = {role: [] for role in ALL_ROLE}

        self.played_cards = {
            'landlord1': [],
            'landlord2': [],
            'landlord3': [],
            'landlord4': []
        }

        self.light_cards = {role: [] for role in ALL_ROLE}

        self.receive_score_cards = {role: [] for role in ALL_ROLE}

        self.last_move = []
        self.last_two_moves = []

        self.info_sets = {
            'landlord1': InfoSet('landlord1'),
            'landlord2': InfoSet('landlord2'),
            'landlord3': InfoSet('landlord3'),
            'landlord4': InfoSet('landlord4'),
        }
        self.last_pid = 'landlord1'
        self.step_count = 0

    def get_infoset(self):
        """
        state, 观测值
        return: 某个具体agent的infoset
        """
        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].last_max_move = self.get_max_by_turn()

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_two_moves = self.get_last_two_moves()

        # 桌子中亮的牌
        self.info_sets[self.acting_player_position].light_cards = self.light_cards

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict  # 暂时不要

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in self.all_role}

        self.info_sets[self.acting_player_position].other_hand_cards = []

        for pos in self.all_role:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards

        # 玩家收到的分牌
        self.info_sets[self.acting_player_position].receive_score_cards = self.receive_score_cards

        # 剩余分牌(公用)
        self.info_sets[self.acting_player_position].remain_score_cards = self.remain_score_cards

        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in self.all_role}

        return deepcopy(self.info_sets[self.acting_player_position])


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """

    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        self.receive_score_cards = None  # 收到的分牌
        self.remain_score_cards = None  # 剩余分牌
        self.curr_score = None  # 当前分
        # The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict = None
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player 
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # 该轮中最大的move
        self.last_max_move = None
        # The most recent valid move
        self.last_move = None  # no obs
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the postions
        self.last_move_dict = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict. 
        self.all_handcards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
        self.light_cards = None  # 亮的牌


infoset = InfoSet("landlord1")
