from collections import Counter
import numpy as np

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}


def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot


def _cards2array_old(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))


def _cards2array(list_cards):
    """
    one_hot编码
    return: shape:(1,52)
    """
    if len(list_cards) == 0:
        return np.zeros(52, dtype=np.int8)
    matrix = np.zeros([4, 13], dtype=np.int8)
    for c in list_cards:
        suit = int(c // 100)
        val = int(c % 100)
        val = val - 2 if suit != 3 else val - 4
        matrix[suit - 1, val] = 1
    # matrix = matrix.flatten('F')  # todo: F按竖的方向降，是否有问题？
    matrix = matrix.flatten()  # default 意思是按行(c型)顺序压平。
    return matrix


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    拱猪一轮4个move，20个move，编码成5x4x52
    """
    action_seq_array = np.zeros((len(action_seq_list), 52))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(10, 4 * 52)  # 5 * 4 * 52, 一轮4个move
    return action_seq_array


def _process_action_seq(sequence, length=20):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _get_obs_new(infoset):
    """
    最新obs
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    # 合法动作数
    num_legal_actions = len(infoset.legal_actions)
    # 手牌
    my_handcards = _cards2array(infoset.player_hand_cards)  # 52
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    # 其它手牌
    other_handcards = _cards2array(infoset.other_hand_cards)  # 52
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    # 该轮此前最大的move 52
    last_max_action = _cards2array(infoset.last_max_move)
    # print("此轮最大的牌：", infoset.last_max_move, last_max_action)
    last_max_action_batch = np.repeat(last_max_action[np.newaxis, :],
                                      num_legal_actions, axis=0)

    # 合法动作
    my_action_batch = np.zeros(my_handcards_batch.shape)  # 52
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    # 1.每名玩家收的分牌
    landlord1_score_cards = _cards2array(
        infoset.receive_score_cards['landlord1'])  # 52
    # print("玩家1收的分：", infoset.receive_score_cards['landlord1'], landlord1_score_cards)
    landlord1_score_cards_batch = np.repeat(
        landlord1_score_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord2_score_cards = _cards2array(
        infoset.receive_score_cards['landlord2'])  # 52
    landlord2_score_cards_batch = np.repeat(
        landlord2_score_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord3_score_cards = _cards2array(
        infoset.receive_score_cards['landlord3'])  # 52
    landlord3_score_cards_batch = np.repeat(
        landlord3_score_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord4_score_cards = _cards2array(
        infoset.receive_score_cards['landlord4'])  # 52
    landlord4_score_cards_batch = np.repeat(
        landlord4_score_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # 2.todo: 剩余分牌？？？ 公共
    remain_score_cards = _cards2array(
        infoset.remain_score_cards)  # 52
    remain_score_cards_batch = np.repeat(
        remain_score_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # 3.其余玩家最新出牌（用于分析玩家手牌分布？）
    # 玩家亮的牌
    landlord1_light_cards = _cards2array(
        infoset.light_cards['landlord1'])  # 52
    landlord1_light_cards_batch = np.repeat(
        landlord1_light_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord2_light_cards = _cards2array(
        infoset.light_cards['landlord2'])  # 52
    landlord2_light_cards_batch = np.repeat(
        landlord2_light_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord3_light_cards = _cards2array(
        infoset.light_cards['landlord3'])  # 52
    landlord3_light_cards_batch = np.repeat(
        landlord3_light_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord4_light_cards = _cards2array(
        infoset.light_cards['landlord4'])  # 52
    landlord4_light_cards_batch = np.repeat(
        landlord4_light_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # 2.每个玩家出过的牌
    landlord1_played_cards = _cards2array(
        infoset.played_cards['landlord1'])  # 52
    landlord1_played_cards_batch = np.repeat(
        landlord1_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord2_played_cards = _cards2array(
        infoset.played_cards['landlord2'])  # 52
    landlord2_played_cards_batch = np.repeat(
        landlord2_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord3_played_cards = _cards2array(
        infoset.played_cards['landlord3'])  # 52
    landlord3_played_cards_batch = np.repeat(
        landlord3_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord4_played_cards = _cards2array(
        infoset.played_cards['landlord4'])  # 52
    landlord4_played_cards_batch = np.repeat(
        landlord4_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,  # 自己的手牌batch
                         other_handcards_batch,  # 其他手牌(总牌 - 已经出的牌 - 自己手牌)
                         last_max_action_batch,  # 该轮最大的action

                         landlord1_light_cards_batch,  # 玩家亮的牌
                         landlord2_light_cards_batch,
                         landlord3_light_cards_batch,
                         landlord4_light_cards_batch,

                         landlord1_played_cards_batch,  # 其余玩家出过的牌
                         landlord2_played_cards_batch,  # 其余玩家出过的牌
                         landlord3_played_cards_batch,
                         landlord4_played_cards_batch,

                         landlord1_score_cards_batch,
                         landlord2_score_cards_batch,  # 其余玩家收的分牌
                         landlord3_score_cards_batch,
                         landlord4_score_cards_batch,
                         remain_score_cards_batch,  # 剩余分牌
                         my_action_batch))

    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_max_action,
                             landlord1_light_cards,
                             landlord2_light_cards,
                             landlord3_light_cards,
                             landlord4_light_cards,

                             landlord1_played_cards,
                             landlord2_played_cards,
                             landlord3_played_cards,
                             landlord4_played_cards,

                             landlord1_score_cards,
                             landlord2_score_cards,
                             landlord3_score_cards,
                             landlord4_score_cards,
                             remain_score_cards,
                             ))
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 40))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def get_obs(infoset):
    """
    """
    if infoset.player_position not in ('landlord1', 'landlord2', 'landlord3', 'landlord4'):
        raise ValueError('')
    return _get_obs_new(infoset)