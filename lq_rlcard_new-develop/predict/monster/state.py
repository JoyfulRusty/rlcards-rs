# -*— coding: utf-8 -*-

import numpy as np

from typing import Callable, Dict
from collections import OrderedDict, Counter

from reinforce.const.monster.const import ActionType, Card2Column, NumOnes2Array
from reinforce.games.monster.utils import new_card_encoding_dict


class ModelState:
    """
    预测状态
    """
    def init_state(self, state):
        """
        初始化model state data
        """
        data = {
            'landlord': state.get("landlord", 0),
            'traces': state.get("traces", []),
            'played_cards': state.get("played_cards", []),
            'self': state.get("self", 0),
            'curr_hand_cards': state.get("curr_hand_cards", []),
            'other_hand_cards': state.get("other_hand_cards", []),
            'all_cards': state.get("all_cards", []),
            'hand_card_nums': state.get("hand_card_nums", []),
            'actions': self.calc_legal_actions(state.get("actions", []), state.get("traces", [])),
            'bust': state.get("bust", [])
        }
        new_state = self.extract_state(data)
        return new_state, data.get("self", 0)

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
        获取obs
        """
        seat_id = state.get("self", 0)
        obs_functions: Dict[int, Callable] = {
            1: self.get_obs_down,
            2: self.get_obs_right,
            3: self.get_obs_up,
            4: self.get_obs_left
        }
        return obs_functions[seat_id](state)

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

    def calc_legal_actions(self, actions, traces):
        """
        计算能够打出的合法动作
        """
        # 判断当前是否能够进行捡牌
        if self.calc_can_picks(traces) and traces[-1][1] != "PICK_CARDS":
            actions.append('PICK_CARDS')
        print("机器人合法操作: ", actions)
        return actions

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
        return np.concatenate((matrix.flatten('F'), magic_matrix))

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
        return np.concatenate((matrix.flatten('F'), magic_matrix))

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
            return matrix.flatten('F')
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
        return matrix.flatten('F')

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

        return matrix.flatten('F')

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
        return np.concatenate((matrix.flatten('F'), magic_matrix))

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
        return np.concatenate((matrix.flatten('F'), magic_matrix))

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
        return matrix.flatten('F')

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
        tu_di_cards, pick_flags = [3, 8, 5], [10, 20]
        for i, trace in enumerate(traces):
            if traces[i][1] == ActionType.PICK_CARDS:
                continue
            # 存在师傅牌则直接跳出
            if (traces[i][1] % 100) == pick_flags[0]:
                return True
            # 未出现师傅牌时，判断万能牌是否作为师傅牌打出
            # 前一张卡牌动作不能捡牌，否则不满足查找条件
            if traces[i - 1][1] != ActionType.PICK_CARDS:
                if (traces[i - 1][1] % 100) in tu_di_cards and (traces[i][1] % 100) in pick_flags:
                    return True
        return False