# -*— coding: utf-8 -*-

from rlcards.games.monster.utils import *

from collections import OrderedDict, Counter
from rlcards.const.monster.const import ActionType, Card2Column, NumOnes2Array


class PredictState:
    """
    预测状态
    """
    def __init__(self):
        """
        初始化参数
        """
        self.bust = []
        self.traces = []
        self.actions = []
        self.new_state = []
        self.encode_state = []
        self.player_id = -1
        self.landlord_id = None
        self.hand_card_nums = []
        self.curr_hand_cards = []
        self.other_hand_cards = [[], [], []]
        self.played_cards = [[], [], [], []]

    def make_state(self, state):
        """
        构建预测状态数据
        """
        self.new_state = self.update_state(state)
        self.encode_state = self.extract_state(self.new_state)

        return self.new_state['self']

    def update_state(self, state):
        """
        更新玩家状态数据
        """
        self.landlord_id = (state['landlord'] - 1)
        self.traces = state['traces']
        self.played_cards = state['played_cards']
        self.player_id = state['self']
        self.curr_hand_cards = state['curr_hand_cards']
        self.other_hand_cards = state['other_hand_cards']
        self.all_cards = state['all_cards']
        self.hand_card_nums = state['hand_card_nums']
        self.actions = self.calc_legal_actions(state)
        self.bust = state['bust']

        new_state = {
            'landlord': self.landlord_id,
            'traces': self.traces,
            'played_cards': self.played_cards,
            'self': self.player_id,
            'curr_hand_cards': self.curr_hand_cards,
            'other_hand_cards': self.other_hand_cards,
            'all_cards': self.all_cards,
            'hand_card_nums': self.hand_card_nums,
            'actions': self.actions,
            'bust': self.bust
        }

        return new_state

    def extract_state(self, state):
        """
        抽取状态编码
        """
        obs, action_history = self.calc_parse_obs(state)
        encode_state = {
            'obs': obs,
            'legal_actions': self.get_legal_actions(state),
            'raw_obs': state,
            'z_obs': obs.reshape(4, 95).astype(np.float32),
            'raw_legal_actions': [a for a in state['actions']],
            'action_history': action_history
        }

        return encode_state

    def calc_parse_obs(self, state):
        """
		选中玩家对应的obs计算流程
		"""
        if state['self'] == 0:
            return self.get_obs_down(state)
        elif state['self'] == 1:
            return self.get_obs_right(state)
        elif state['self'] == 2:
            return self.get_obs_up(state)
        return self.get_obs_left(state)

    def get_obs_down(self, state):
        """
		下家[0]
		"""
        # 下家手牌
        down_curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
        down_played_cards = self.encode_played_cards(state['played_cards']['down'], state['bust']['down'])

        right_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['right'], state['bust']['right'])
        right_played_cards = self.encode_played_cards(state['played_cards']['right'], state['bust']['right'])
        right_bust = self.encode_bust(state['bust']['right'])

        up_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['up'], state['bust']['up'])
        up_played_cards = self.encode_played_cards(state['played_cards']['up'], state['bust']['up'])
        up_bust = self.encode_bust(state['bust']['up'])

        left_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['left'], state['bust']['left'])
        left_player_cards = self.encode_played_cards(state['played_cards']['left'], state['bust']['left'])
        left_bust = self.encode_bust(state['bust']['left'])

        all_cards = self.encode_all_cards(state['all_cards'])

        legal_actions = self.encode_legal_actions(state['actions'])
        last_action = self.encode_last_action(state['traces'])
        before_pick_action = self.encode_before_pick_action(state['traces'])
        action_history = self.action_seq_history(state['traces'])

        obs = np.hstack((
            down_curr_hand_cards,
            down_played_cards,
            right_hand_cards,
            right_played_cards,
            right_bust,
            up_hand_cards,
            up_played_cards,
            up_bust,
            left_hand_cards,
            left_player_cards,
            left_bust,
            all_cards,
            legal_actions,
            last_action,
            before_pick_action,
            action_history
        ))

        return obs, action_history

    def get_obs_right(self, state):
        """
		右家[1]
		"""
        right_curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
        right_played_cards = self.encode_played_cards(state['played_cards']['right'], state['bust']['right'])

        up_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['up'], state['bust']['up'])
        up_played_cards = self.encode_played_cards(state['played_cards']['up'], state['bust']['up'])
        up_bust = self.encode_bust(state['bust']['up'])

        left_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['left'], state['bust']['left'])
        left_played_cards = self.encode_played_cards(state['played_cards']['left'], state['bust']['left'])
        left_bust = self.encode_bust(state['bust']['left'])

        down_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['down'], state['bust']['down'])
        down_played_cards = self.encode_played_cards(state['played_cards']['down'], state['bust']['down'])
        down_bust = self.encode_bust(state['bust']['down'])

        all_cards = self.encode_all_cards(state['all_cards'])

        legal_actions = self.encode_legal_actions(state['actions'])
        last_action = self.encode_last_action(state['traces'])
        before_pick_action = self.encode_before_pick_action(state['traces'])
        action_history = self.action_seq_history(state['traces'])

        obs = np.hstack((
            right_curr_hand_cards,
            right_played_cards,
            up_hand_cards,
            up_played_cards,
            up_bust,
            left_hand_cards,
            left_played_cards,
            left_bust,
            down_hand_cards,
            down_played_cards,
            down_bust,
            all_cards,
            legal_actions,
            last_action,
            before_pick_action,
            action_history
        ))

        return obs, action_history

    def get_obs_up(self, state):
        """
		上家[2]
		"""
        up_curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
        up_played_cards = self.encode_played_cards(state['played_cards']['up'], state['bust']['up'])

        left_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['left'], state['bust']['left'])
        left_played_cards = self.encode_played_cards(state['played_cards']['left'], state['bust']['left'])
        left_bust = self.encode_bust(state['bust']['left'])

        down_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['down'], state['bust']['down'])
        down_played_cards = self.encode_played_cards(state['played_cards']['down'], state['bust']['down'])
        down_bust = self.encode_bust(state['bust']['down'])

        right_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['right'], state['bust']['right'])
        right_played_cards = self.encode_played_cards(state['played_cards']['right'], state['bust']['right'])
        right_bust = self.encode_bust(state['bust']['right'])

        all_cards = self.encode_all_cards(state['all_cards'])

        legal_actions = self.encode_legal_actions(state['actions'])
        last_action = self.encode_last_action(state['traces'])
        before_pick_action = self.encode_before_pick_action(state['traces'])
        action_history = self.action_seq_history(state['traces'])

        obs = np.hstack((
            up_curr_hand_cards,
            up_played_cards,
            left_hand_cards,
            left_played_cards,
            left_bust,
            down_hand_cards,
            down_played_cards,
            down_bust,
            right_hand_cards,
            right_played_cards,
            right_bust,
            all_cards,
            legal_actions,
            last_action,
            before_pick_action,
            action_history
        ))

        return obs, action_history

    def get_obs_left(self, state):
        """
		左家[3]
		"""
        left_curr_hand_cards = self.encode_cards(state['curr_hand_cards'])
        left_played_cards = self.encode_played_cards(state['played_cards']['left'], state['bust']['left'])

        down_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['down'], state['bust']['down'])
        down_played_cards = self.encode_played_cards(state['played_cards']['down'], state['bust']['down'])
        down_bust = self.encode_bust(state['bust']['down'])

        right_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['right'], state['bust']['right'])
        right_played_cards = self.encode_played_cards(state['played_cards']['right'], state['bust']['right'])
        right_bust = self.encode_bust(state['bust']['right'])

        up_hand_cards = self.encode_other_hand_cards(state['other_hand_cards']['up'], state['bust']['up'])
        up_played_cards = self.encode_played_cards(state['played_cards']['up'], state['bust']['up'])
        up_bust = self.encode_bust(state['bust']['up'])

        all_cards = self.encode_all_cards(state['all_cards'])

        legal_actions = self.encode_legal_actions(state['actions'])
        last_action = self.encode_last_action(state['traces'])
        before_pick_action = self.encode_before_pick_action(state['traces'])
        action_history = self.action_seq_history(state['traces'])

        obs = np.hstack((
            left_curr_hand_cards,
            left_played_cards,
            down_hand_cards,
            down_played_cards,
            down_bust,
            right_hand_cards,
            right_played_cards,
            right_bust,
            up_hand_cards,
            up_played_cards,
            up_bust,
            all_cards,
            legal_actions,
            last_action,
            before_pick_action,
            action_history
        ))

        return obs, action_history

    def calc_legal_actions(self, state):
        """
        计算能够打出的合法动作
        """
        # 判断当前是否能够进行捡牌
        if self.calc_can_picks(state['traces']) and self.traces[-1][1] != "PICK_CARDS":
            legal_actions = state['actions']
            legal_actions.append('PICK_CARDS')
        # 当前不满足捡牌条件[只能进行正常出牌]
        else:
            legal_actions = state['actions']
        print("当前合法动作为: ", legal_actions)
        return legal_actions

    @staticmethod
    def encode_cards(cards):
        """
		卡牌编码
		1: 'D',  # 方块♦
		2: 'C',  # 梅花♣
		3: 'H',  # 红心♥
		4: 'S',  # 黑桃♠
		"""
        if not cards:
            return np.zeros(32, dtype=np.float32)
        matrix = np.zeros((4, 7), dtype=np.float32)
        magic_matrix = np.zeros(4, dtype=np.float32)
        count_magic = 0
        for card in cards:
            idx = (card // 100) - 1
            if idx == 4:
                count_magic += 1
                continue
            matrix[idx][Card2Column[card % 100]] = 1
        if count_magic > 0:
            magic_matrix[:count_magic] = 1
        return np.concatenate((matrix.flatten('A'), magic_matrix))

    @staticmethod
    def encode_other_hand_cards(other_cards, bust):
        """
		编码其他玩家手牌
		"""
        if not other_cards or bust:
            return np.zeros(32, dtype=np.float32)
        matrix = np.zeros((4, 7), dtype=np.float32)
        magic_matrix = np.zeros(4, dtype=np.float32)
        count_magic = 0
        for card in other_cards:
            idx = (card // 100) - 1
            if idx == 4:
                count_magic += 1
                continue
            matrix[idx][Card2Column[card % 100]] = 1
        if count_magic > 0:
            magic_matrix[:count_magic] = 1
        return np.concatenate((matrix.flatten('A'), magic_matrix))

    @staticmethod
    def encode_bust(bust):
        """
		编码破产玩家
		"""
        if not bust:
            # 未破产为全1
            return np.ones(1, dtype=np.float32)
        # 破产为全0
        return np.zeros(1, dtype=np.float32)

    @staticmethod
    def encode_before_pick_action(traces):
        """
		编码导致捡牌的卡牌
		"""
        if len(traces) < 2:
            return np.zeros(8, dtype=np.float32)
        if traces[-1][1] == ActionType.PICK_CARDS:
            matrix = np.zeros(8, dtype=np.float32)
            # 编码导致捡牌的卡牌
            matrix[Card2Column[traces[-2][1] % 100]] = 1
            return matrix.flatten('A')
        return np.zeros(8, dtype=np.float32)

    @staticmethod
    def encode_last_action(traces):
        """
		编码上一个动作
		"""
        if not traces:
            return np.zeros(9, dtype=np.float32)
        matrix = np.zeros((1, 9), dtype=np.float32)
        if traces[-1][1] == ActionType.PICK_CARDS:
            matrix[:, Card2Column[traces[-1][1]]] = 1
        else:
            matrix[:, Card2Column[traces[-1][1] % 100]] = 1
        return matrix.flatten('A')

    @staticmethod
    def encode_legal_actions(actions):
        """
		编码合法动作
		"""
        if not actions:
            return np.zeros(36, dtype=np.float32)
        matrix = np.zeros((4, 9), dtype=np.float32)
        actions_dict = Counter(actions)
        for action, nums in actions_dict.items():
            if action == ActionType.PICK_CARDS:
                matrix[:, Card2Column[action]] = NumOnes2Array[nums]
            else:
                matrix[:, Card2Column[action % 100]] = NumOnes2Array[nums]

        return matrix.flatten('A')

    @staticmethod
    def encode_played_cards(played_cards, bust):
        """
		编码玩家打出的牌
		"""
        if not played_cards or bust:
            return np.zeros(32, dtype=np.float32)
        matrix = np.zeros((4, 7), dtype=np.float32)
        magic_matrix = np.zeros(4, dtype=np.float32)
        count_magic = 0
        for card in played_cards:
            idx = (card // 100) - 1
            if idx == 4:
                count_magic += 1
                continue
            matrix[idx][Card2Column[card % 100]] = 1
        if count_magic > 0:
            magic_matrix[:count_magic] = 1
        return np.concatenate((matrix.flatten('A'), magic_matrix))

    @staticmethod
    def encode_all_cards(all_cards):
        """
		编码所有卡牌
		"""
        if not all_cards:
            return np.zeros(32, dtype=np.float32)
        matrix = np.zeros((4, 7), dtype=np.float32)
        magic_matrix = np.zeros(4, dtype=np.float32)
        count_magic = 0
        for card in all_cards:
            idx = (card // 100) - 1
            if idx == 4:
                count_magic += 1
                continue
            matrix[idx][Card2Column[card % 100]] = 1
        if count_magic > 0:
            magic_matrix[:count_magic] = 1
        return np.concatenate((matrix.flatten('A'), magic_matrix))

    @staticmethod
    def action_seq_history(action_seqs):
        """
		TODO: 玩家打牌动作序列
		只对玩家动作序列的后四个动作进行编码
		以四个动作为一个滑动窗口移动编码
		"""
        if not action_seqs:
            return np.zeros(36, dtype=np.float32)
        matrix = np.zeros((4, 9), dtype=np.float32)
        for cards in action_seqs[-4:]:
            if cards[1] == ActionType.PICK_CARDS:
                matrix[cards[0]][Card2Column[cards[1]]] = 1
            else:
                matrix[cards[0]][Card2Column[cards[1] % 100]] = 1

        return matrix.flatten('A')

    @staticmethod
    def get_legal_actions(state):
        """
        合法动作
        """
        legal_action_id = {}
        legal_actions = state['actions']
        if legal_actions:
            for action in legal_actions:
                legal_action_id[new_card_encoding_dict[action]] = None

        return OrderedDict(legal_action_id)

    @staticmethod
    def calc_can_picks(traces):
        """
        判断首次捡牌是否符合规则
        """
        tu_di_cards = [3, 8, 5]
        pick_flag = [10, 20]
        for i, trace in enumerate(traces):
            # 无师傅牌，则判断万能牌是否作为师傅牌
            if traces[i][1] == ActionType.PICK_CARDS:
                continue
            # 存在师傅牌则直接跳出
            if (traces[i][1] % 100) == pick_flag[0]:
                return True
            # 未出现师傅牌时，判断万能牌是否作为师傅牌打出
            # 前一张卡牌动作不能捡牌，否则不满足查找条件
            if traces[i - 1][1] != ActionType.PICK_CARDS:
                if (traces[i - 1][1] % 100) in tu_di_cards and (traces[i][1] % 100) in pick_flag:
                    return True

# 封装状态数据
predict_state = PredictState()