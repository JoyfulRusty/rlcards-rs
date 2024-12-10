import numpy as np

from predict.ddz.xxc.const import B_KING, S_KING

FK_3 = 31

def get_obs_by_4_an(info_set):
    if info_set.player_position == 'landlord1':
        return _get_obs_landlord1(info_set)
    elif info_set.player_position == 'landlord2':
        return _get_obs_landlord2(info_set)
    elif info_set.player_position == 'landlord3':
        return _get_obs_landlord3(info_set)
    elif info_set.player_position == 'landlord4':
        return _get_obs_landlord4(info_set)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot encoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1
    return one_hot

def _cards2array(list_cards, played_3=False, desc=""):
    """
    one_hot编码
    需要注意地主牌方片3的花色，在此编码时考虑进去
    return: shape:(1,52)
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)  # 大小王
    matrix = np.zeros([4, 13], dtype=np.int8)
    exist_curr = played_3 and FK_3 not in list_cards
    for c in list_cards:
        val = int(c % 100)
        if val == S_KING:
            jokers[0] = 1
        elif val == B_KING:
            jokers[1] = 1
        elif (val == FK_3 or (val == 3 and exist_curr)) and matrix[0, 0] == 0:
            matrix[0, 0] = 1  # 方块3
        else:
            if val > 16:
                raise ValueError("val error", val, "list_cards: ", list_cards)
            if val == 16:
                val -= 5
            else:
                val -= 3
            if val == 0:
                if matrix[1, val] != 1:
                    matrix[1, val] = 1
                elif matrix[2, val] != 1:
                    matrix[2, val] = 1
                elif matrix[3, val] != 1:
                    matrix[3, val] = 1
            else:
                if matrix[0, val] != 1:
                    matrix[0, val] = 1
                elif matrix[1, val] != 1:
                    matrix[1, val] = 1
                elif matrix[2, val] != 1:
                    matrix[2, val] = 1
                elif matrix[3, val] != 1:
                    matrix[3, val] = 1
    return np.concatenate((matrix.flatten('F'), jokers))


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDi zhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    一轮4个move, 5 x 4 x 54
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards, FK_3 in list_cards, "action_seq")
    action_seq_array = action_seq_array.reshape(5, 216)
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

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord1(info_set):
    """
    地主1视角的观测
    """
    is_landlord_3 = info_set.has_fk_3_di_zhu
    played_fk_3 = info_set.played_fk_3
    record = not played_fk_3 and is_landlord_3  # 方块3没出并且是有方块3的地主
    other_record = not played_fk_3 and not is_landlord_3  # 方块三没出并且不是有方块3的地主(用作其他手牌)

    num_legal_actions = len(info_set.legal_actions)
    my_hand_cards = _cards2array(info_set.player_hand_cards, record, "land1手牌")  # 手牌
    my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    other_hand_cards = _cards2array(info_set.other_hand_cards, other_record, "除land1外其他手牌")  # 其他手牌
    other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move, FK_3 in info_set.last_move, "上一个动作")  # 上一个动作
    last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_hand_cards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action, FK_3 in action, "合法动作")  # 合法动作

    # landlord2上一个动作
    land2_last_action = info_set.last_move_dict['landlord2']
    last_landlord2_action = _cards2array(
        land2_last_action, FK_3 in land2_last_action, "land2上一个动作")
    last_landlord2_action_batch = np.repeat(
        last_landlord2_action[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2剩余牌数
    landlord2_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord2'], 13)
    landlord2_num_cards_left_batch = np.repeat(
        landlord2_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2打过的牌
    land2_played_cards = info_set.played_cards['landlord2']
    landlord2_played_cards = _cards2array(
        land2_played_cards, FK_3 in land2_played_cards, "land2打过的牌")
    landlord2_played_cards_batch = np.repeat(
        landlord2_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    land3_last_action = info_set.last_move_dict['landlord3']
    last_landlord3_action = _cards2array(
        land3_last_action, FK_3 in land3_last_action, "land3上一个动作")
    last_landlord3_action_batch = np.repeat(
        last_landlord3_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord3_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord3'], 13)
    landlord3_num_cards_left_batch = np.repeat(
        landlord3_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land3_played_cards = info_set.played_cards['landlord3']
    landlord3_played_cards = _cards2array(
        land3_played_cards, FK_3 in land3_played_cards, "land3打过的牌")
    landlord3_played_cards_batch = np.repeat(
        landlord3_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    land4_last_action = info_set.last_move_dict['landlord4']
    last_landlord4_action = _cards2array(
        land4_last_action, FK_3 in land4_last_action, "land4上一个动作")
    last_landlord4_action_batch = np.repeat(
        last_landlord4_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord4_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord4'], 13)
    landlord4_num_cards_left_batch = np.repeat(
        landlord4_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land4_played_cards = info_set.played_cards['landlord4']
    landlord4_played_cards = _cards2array(
        land4_played_cards, FK_3 in land4_played_cards, "land4打过的牌")
    landlord4_played_cards_batch = np.repeat(
        landlord4_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        info_set.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_hand_cards_batch,  # 54
                         other_hand_cards_batch,  # 54
                         last_action_batch,  # 54
                         last_landlord2_action_batch,  # 54
                         last_landlord3_action_batch,  # 54
                         last_landlord4_action_batch,  # 54

                         landlord2_played_cards_batch,  # 54
                         landlord3_played_cards_batch,  # 54
                         landlord4_played_cards_batch,  # 54

                         landlord2_num_cards_left_batch,  # 13
                         landlord3_num_cards_left_batch,  # 13
                         landlord4_num_cards_left_batch,  # 13
                         bomb_num_batch,  # 15
                         my_action_batch))  # 54
    x_no_action = np.hstack((my_hand_cards,
                             other_hand_cards,
                             last_action,
                             last_landlord2_action,
                             last_landlord3_action,
                             last_landlord4_action,

                             landlord2_played_cards,
                             landlord3_played_cards,
                             landlord4_played_cards,

                             landlord2_num_cards_left,
                             landlord3_num_cards_left,
                             landlord4_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        info_set.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    # print("x_batch1: ", x_batch.shape)
    obs = {
        'position': 'landlord1',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': info_set.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_obs_landlord2(info_set):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    is_landlord_3 = info_set.has_fk_3_di_zhu
    played_fk_3 = info_set.played_fk_3
    record = not played_fk_3 and is_landlord_3  # 方块3没出并且是有方块3的地主
    other_record = not played_fk_3 and not is_landlord_3  # 方块三没出并且不是有方块3的地主(用作其他手牌)

    num_legal_actions = len(info_set.legal_actions)
    my_hand_cards = _cards2array(info_set.player_hand_cards, record, "land2手牌")  # 手牌
    my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    other_hand_cards = _cards2array(info_set.other_hand_cards, other_record, "除land2外其他手牌")  # 其他手牌
    other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move, FK_3 in info_set.last_move, "上一个动作")  # 上一个动作
    last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_hand_cards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action, FK_3 in action, "合法动作")  # 合法动作

    # landlord1上一个动作
    land1_last_action = info_set.last_move_dict['landlord1']
    last_landlord1_action = _cards2array(
        land1_last_action, FK_3 in land1_last_action, "land1上一个动作")
    last_landlord1_action_batch = np.repeat(
        last_landlord1_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord1_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord1'], 15)
    landlord1_num_cards_left_batch = np.repeat(
        landlord1_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land1_played_cards = info_set.played_cards['landlord1']
    landlord1_played_cards = _cards2array(
        land1_played_cards, FK_3 in land1_played_cards, "land1打过的牌")
    landlord1_played_cards_batch = np.repeat(
        landlord1_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    land3_last_action = info_set.last_move_dict['landlord3']
    last_landlord3_action = _cards2array(
        land3_last_action, FK_3 in land3_last_action, "land3上一个动作")
    last_landlord3_action_batch = np.repeat(
        last_landlord3_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord3_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord3'], 13)
    landlord3_num_cards_left_batch = np.repeat(
        landlord3_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land3_played_cards = info_set.played_cards['landlord3']
    landlord3_played_cards = _cards2array(
        land3_played_cards, FK_3 in land3_played_cards, "land3打过的牌")
    landlord3_played_cards_batch = np.repeat(
        landlord3_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    land4_last_action = info_set.last_move_dict['landlord4']
    last_landlord4_action = _cards2array(
        land4_last_action, FK_3 in land4_last_action, "land4上一个动作")
    last_landlord4_action_batch = np.repeat(
        last_landlord4_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord4_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord4'], 13)
    landlord4_num_cards_left_batch = np.repeat(
        landlord4_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land4_played_cards = info_set.played_cards['landlord4']
    landlord4_played_cards = _cards2array(
        land4_played_cards, FK_3 in land4_played_cards, "land4打过的牌")
    landlord4_played_cards_batch = np.repeat(
        landlord4_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        info_set.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_hand_cards_batch,
                         other_hand_cards_batch,
                         last_action_batch,
                         last_landlord1_action_batch,
                         last_landlord3_action_batch,
                         last_landlord4_action_batch,
                         landlord1_played_cards_batch,
                         landlord3_played_cards_batch,
                         landlord4_played_cards_batch,
                         landlord1_num_cards_left_batch,
                         landlord3_num_cards_left_batch,
                         landlord4_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_hand_cards,
                             other_hand_cards,
                             last_action,
                             last_landlord1_action,
                             last_landlord3_action,
                             last_landlord4_action,
                             landlord1_played_cards,
                             landlord3_played_cards,
                             landlord4_played_cards,
                             landlord1_num_cards_left,
                             landlord3_num_cards_left,
                             landlord4_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        info_set.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    # print("x_batch2: ", x_batch.shape)
    obs = {
        'position': 'landlord2',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': info_set.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_obs_landlord3(info_set):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    is_landlord_3 = info_set.has_fk_3_di_zhu
    played_fk_3 = info_set.played_fk_3
    record = not played_fk_3 and is_landlord_3  # 方块3没出并且是有方块3的地主
    other_record = not played_fk_3 and not is_landlord_3  # 方块三没出并且不是有方块3的地主(用作其他手牌)

    num_legal_actions = len(info_set.legal_actions)
    my_hand_cards = _cards2array(info_set.player_hand_cards, record, "land3手牌")  # 手牌
    my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    other_hand_cards = _cards2array(info_set.other_hand_cards, other_record, "除land3外其他手牌")  # 其他手牌
    other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move, FK_3 in info_set.last_move, "上一个动作")  # 上一个动作
    last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_hand_cards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action, FK_3 in action, "合法动作")  # 合法动作

    land1_last_action = info_set.last_move_dict['landlord1']
    last_landlord1_action = _cards2array(
        land1_last_action, FK_3 in land1_last_action, "land1上一个动作")
    last_landlord1_action_batch = np.repeat(
        last_landlord1_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord1_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord1'], 15)
    landlord1_num_cards_left_batch = np.repeat(
        landlord1_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land1_played_cards = info_set.played_cards['landlord1']
    landlord1_played_cards = _cards2array(
        land1_played_cards, FK_3 in land1_played_cards, "land1打过的牌")
    landlord1_played_cards_batch = np.repeat(
        landlord1_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2上一个动作
    land2_last_action = info_set.last_move_dict['landlord2']
    last_landlord2_action = _cards2array(
        land2_last_action, FK_3 in land2_last_action, "land2上一个动作")
    last_landlord2_action_batch = np.repeat(
        last_landlord2_action[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2剩余牌数
    landlord2_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord2'], 13)
    landlord2_num_cards_left_batch = np.repeat(
        landlord2_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2打过的牌
    land2_played_cards = info_set.played_cards['landlord2']
    landlord2_played_cards = _cards2array(
        land2_played_cards, FK_3 in land2_played_cards, "land2打过的牌")
    landlord2_played_cards_batch = np.repeat(
        landlord2_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    land4_last_action = info_set.last_move_dict['landlord4']
    last_landlord4_action = _cards2array(
        land4_last_action, FK_3 in land4_last_action, "land4上一个动作")
    last_landlord4_action_batch = np.repeat(
        last_landlord4_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord4_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord4'], 13)
    landlord4_num_cards_left_batch = np.repeat(
        landlord4_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land4_played_cards = info_set.played_cards['landlord4']
    landlord4_played_cards = _cards2array(
        land4_played_cards, FK_3 in land4_played_cards, "land4大过的牌")
    landlord4_played_cards_batch = np.repeat(
        landlord4_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        info_set.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_hand_cards_batch,
                         other_hand_cards_batch,
                         last_action_batch,
                         last_landlord1_action_batch,
                         last_landlord2_action_batch,
                         last_landlord4_action_batch,
                         landlord1_played_cards_batch,
                         landlord2_played_cards_batch,
                         landlord4_played_cards_batch,
                         landlord1_num_cards_left_batch,
                         landlord2_num_cards_left_batch,
                         landlord4_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_hand_cards,
                             other_hand_cards,
                             last_action,
                             last_landlord1_action,
                             last_landlord2_action,
                             last_landlord4_action,
                             landlord1_played_cards,
                             landlord2_played_cards,
                             landlord4_played_cards,
                             landlord1_num_cards_left,
                             landlord2_num_cards_left,
                             landlord4_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        info_set.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    # print("x_batch3: ", x_batch.shape)
    obs = {
        'position': 'landlord3',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': info_set.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_obs_landlord4(info_set):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    is_landlord_3 = info_set.has_fk_3_di_zhu
    played_fk_3 = info_set.played_fk_3
    record = not played_fk_3 and is_landlord_3  # 方块3没出并且是有方块3的地主
    other_record = not played_fk_3 and not is_landlord_3  # 方块三没出并且不是有方块3的地主(用作其他手牌)

    num_legal_actions = len(info_set.legal_actions)
    my_hand_cards = _cards2array(info_set.player_hand_cards, record, "land4手牌")  # 手牌
    my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    other_hand_cards = _cards2array(info_set.other_hand_cards, other_record, "除land4外的手牌")  # 其他手牌
    other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :], num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move, FK_3 in info_set.last_move, "上一个动作")  # 上一个动作
    last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_hand_cards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action, FK_3 in action, "合法动作")  # 合法动作

    land1_last_action = info_set.last_move_dict['landlord1']
    last_landlord1_action = _cards2array(
        land1_last_action, FK_3 in land1_last_action, "land1上一个动作")
    last_landlord1_action_batch = np.repeat(
        last_landlord1_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord1_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord1'], 15)
    landlord1_num_cards_left_batch = np.repeat(
        landlord1_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land1_played_cards = info_set.played_cards['landlord1']
    landlord1_played_cards = _cards2array(
        land1_played_cards, FK_3 in land1_played_cards, "land1打过的牌")
    landlord1_played_cards_batch = np.repeat(
        landlord1_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # land2上一个动作
    land2_last_action = info_set.last_move_dict['landlord2']
    last_landlord2_action = _cards2array(
        land2_last_action, FK_3 in land2_last_action, "land2上一个动作")
    last_landlord2_action_batch = np.repeat(
        last_landlord2_action[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2剩余牌数
    landlord2_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord2'], 13)
    landlord2_num_cards_left_batch = np.repeat(
        landlord2_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    # landlord2打过的牌
    land2_played_cards = info_set.played_cards['landlord2']
    landlord2_played_cards = _cards2array(
        land2_played_cards, FK_3 in land2_played_cards, "land2打过的牌")
    landlord2_played_cards_batch = np.repeat(
        landlord2_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    land3_last_action = info_set.last_move_dict['landlord3']
    last_landlord3_action = _cards2array(
        land3_last_action, FK_3 in land3_last_action, "land3上一个动作")
    last_landlord3_action_batch = np.repeat(
        last_landlord3_action[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord3_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord3'], 13)
    landlord3_num_cards_left_batch = np.repeat(
        landlord3_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    land3_played_cards = info_set.played_cards['landlord3']
    landlord3_played_cards = _cards2array(
        land3_played_cards, FK_3 in land3_played_cards, "land3打过的牌")
    landlord3_played_cards_batch = np.repeat(
        landlord3_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        info_set.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_hand_cards_batch,
                         other_hand_cards_batch,
                         last_action_batch,
                         last_landlord1_action_batch,
                         last_landlord2_action_batch,
                         last_landlord3_action_batch,
                         landlord1_played_cards_batch,
                         landlord2_played_cards_batch,
                         landlord3_played_cards_batch,
                         landlord1_num_cards_left_batch,
                         landlord2_num_cards_left_batch,
                         landlord3_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_hand_cards,
                             other_hand_cards,
                             last_action,
                             last_landlord1_action,
                             last_landlord2_action,
                             last_landlord3_action,
                             landlord1_played_cards,
                             landlord2_played_cards,
                             landlord3_played_cards,
                             landlord1_num_cards_left,
                             landlord2_num_cards_left,
                             landlord3_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        info_set.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    # print("x_batch4: ", x_batch.shape)
    obs = {
        'position': 'landlord4',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': info_set.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs