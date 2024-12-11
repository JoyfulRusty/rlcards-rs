from collections import Counter, OrderedDict
import numpy as np

from rlcards.envs import Env
from rlcards.data.doudizhu.read_data import ACTION_2_ID, ID_2_ACTION
from rlcards.games.doudizhu.utils import cards2str, cards2str_with_suit
from rlcards.games.doudizhu.game import DdzGame as Game

class DdzEnv(Env):
    """
    斗地主环境
    """
    def __init__(self, config):
        self.name = 'doudizhu'
        self.game = Game()
        super().__init__(config, self.game)

        self._cards2str = cards2str
        self._cards2str_with_suit = cards2str_with_suit
        self._ACTION_2_ID = ACTION_2_ID
        self._ID_2_ACTION = ID_2_ACTION
        self.state_shape = [[790], [901], [901]]
        self.action_shape = [[54] for _ in range(self.num_players)]

    def extract_state(self, state):
        """
        编码状态
        :param state(dict): 原始状态的字典
        :return: 编码状态字典
        """
        current_hand = _cards2array(state['current_hand'])
        others_hand = _cards2array(state['others_hand'])

        last_action = ''
        if len(state['trace']) != 0:
            if state['trace'][-1][1] == 'pass':
                last_action = state['trace'][-2][1]
            else:
                last_action = state['trace'][-1][1]
        last_action = _cards2array(last_action)

        last_9_actions = _action_seq2array(_process_action_seq(state['trace']))

        if state['self'] == 0:  # landlord
            landlord_up_played_cards = _cards2array(state['played_cards'][2])
            landlord_down_played_cards = _cards2array(state['played_cards'][1])
            landlord_up_num_cards_left = _get_one_hot_array(state['num_cards_left'][2], 17) 
            landlord_down_num_cards_left = _get_one_hot_array(state['num_cards_left'][1], 17)
            obs = np.concatenate((current_hand,
                                  others_hand,
                                  last_action,
                                  last_9_actions,
                                  landlord_up_played_cards,
                                  landlord_down_played_cards,
                                  landlord_up_num_cards_left,
                                  landlord_down_num_cards_left))
        else:
            landlord_played_cards = _cards2array(state['played_cards'][0])
            for i, action in reversed(state['trace']):
                if i == 0:
                    last_landlord_action = action
            last_landlord_action = _cards2array(last_landlord_action)
            landlord_num_cards_left = _get_one_hot_array(state['num_cards_left'][0], 20)

            teammate_id = 3 - state['self']
            teammate_played_cards = _cards2array(state['played_cards'][teammate_id])
            last_teammate_action = 'pass'
            for i, action in reversed(state['trace']):
                if i == teammate_id:
                    last_teammate_action = action
            last_teammate_action = _cards2array(last_teammate_action)
            teammate_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_id], 17)
            obs = np.concatenate((current_hand,
                                  others_hand,
                                  last_action,
                                  last_9_actions,
                                  landlord_played_cards,
                                  teammate_played_cards,
                                  last_landlord_action,
                                  last_teammate_action,
                                  landlord_num_cards_left,
                                  teammate_num_cards_left))

        extracted_state = OrderedDict({'obs': obs, 'legal_actions': self._get_legal_actions()})
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):
        """
        获得玩家的收益
        :return: 每个玩家的收益列表
        """
        return self.game.judge.judge_payoffs(self.game.round.landlord_id, self.game.winner_id)

    def _decode_action(self, action_id):
        """
        解码动作
        :param action_id: 动作id
        """
        return self._ID_2_ACTION[action_id]

    def _get_legal_actions(self):
        """
        获取当前状态的所有法律行为
        :return: legal_actions(list):法律行为的id列表
        """
        legal_actions = self.game.state['actions']
        legal_actions = {self._ACTION_2_ID[action]: _cards2array(action) for action in legal_actions}
        return legal_actions

    def get_perfect_information(self):
        """
        获取当前状态的完美信息
        :return: (dict)当前状态所有完美信息的字典
        """
        state = {}
        state['hand_cards_with_suit'] = [self._cards2str_with_suit(player.current_hand) for player in self.game.players]
        state['hand_cards'] = [self._cards2str(player.current_hand) for player in self.game.players]
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state

    def get_action_feature(self, action):
        """
        对于一些环境比如斗地主，可以有动作特征
        :param action: 动作
        :return: (numpy.array) -> 动作特点
        """
        return _cards2array(self._decode_action(action))

Card2Column = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7,
               'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def _cards2array(cards):
    if cards == 'pass' or len(cards) >= 2:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(cards)
    for card, num_times in counter.items():
        if card == 'B':
            jokers[0] = 1
        elif card == 'R':
            jokers[1] = 1
        else:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
    # 'F' 按竖的方向降
    return np.concatenate((matrix.flatten('F'), jokers))

def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards, dtype=np.int8)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _action_seq2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54), np.int8)
    for row, cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(cards)
    action_seq_array = action_seq_array.flatten()
    return action_seq_array

def _process_action_seq(sequence, length=9):
    sequence = [action[1] for action in sequence[-length:]]
    if len(sequence) < length:
        empty_sequence = ['' for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence