from collections import Counter
import numpy as np

from ..env.game import GameEnv

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    斗地主多智能包装器
    """
    def __init__(self, objective):
        """
        目标是wp/adp/logadp。表示是否考虑奖励计算中的炸弹。
        使用伪代理这是因为，在最初的游戏中，玩家正在比赛中。
        想要隔离球员和环境更具健身房风格界面为了实现这一点，使用虚拟玩家玩。
        对于每一个动作，都会告诉相应的模拟玩家要玩哪个动作，然后是玩家将在游戏引擎中执行实际操作
        """
        self.objective = objective

        # 初始化玩家
        # 使用三个假人作为目标位置
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)

        # 初始化内部环境
        self._env = GameEnv(self.players)

        self.info_set = None

    def reset(self):
        """
        每次调用重置时，环境将用一副新的牌重新初始化。
        此函数通常在游戏结束时调用。
        """
        self._env.reset()

        #随机打乱卡牌
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20],
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        # 初始化卡牌
        self._env.card_play_init(card_play_data)
        self.info_set = self._game_info_set

        return get_obs(self.info_set)

    def step(self, action):
        """
        Step函数将动作作为输入是整数列表，并输出下一个观测值、奖励、以及指示当前游戏结束。
        它还返回一个空为传递有用信息而保留的词典
        """
        assert action in self.info_set.legal_actions
        self.players[self._acting_player_position].set_action(action)
        # base step
        self._env.step()
        self.info_set = self._game_info_set
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.info_set)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        此函数在每个函数的末尾调用游戏它为输赢返回1/-1，或ADP，即每枚炸弹的得分都会翻倍
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    @property
    def _game_info_set(self):
        """
        这里，信息集被定义为所有信息在目前的情况下，包括手牌所有的玩家，所有的历史动作，等等。
        也就是说，它包含了完美的信息，后来将使用函数来提取可观测的来自三名球员观点的信息。
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        到目前为止播放的炸弹数量。这被用作神经网络的一个特征，也用于计算ADP
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """
        获取获胜玩家
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        处于活动状态的玩家[landlord, landlord_down, landlord_up]
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """
        游戏是否结束
        """
        return self._env.game_over

class DummyAgent(object):
    """
    虚拟代理被设计为易于与游戏引擎，代理人将首先被告知采取什么行动执行。
    然后环境将调用此代理以执行实际动作。这可以帮助将环境和代理隔离到健身房界面。
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, info_set):
        """
        只需返回之前设置的操作即可
        """
        assert self.action in info_set.legal_actions
        return self.action

    def set_action(self, action):
        """
        环境使用此功能来告知伪代理该做什么
        """
        self.action = action

def get_obs(info_set):
    """
    此函数用于获得信息不完善的观测值来自info_set，自从编码以来，它有三个分支不同位置的不同特征。
    将返回名为“obs”的字典，它包含几个领域，这些字段将用于训练模型，人们可以利用这些功能来提高性能。
    :param position: [landlord, landlord_down, landlord_up]
    :param x_batch: 不包括历史信息移动，对动作特征进行编码
    :param legal_actions: 合法动作
    :param x_no_action: 特征(不包括history, movements和动作特征)，没有批次
    :param z_batch: 只记录连续出牌动作移动的功能
    :param z: 与z_batch相同，但不是一个批次
    """
    if info_set.player_position == 'landlord':
        return _get_obs_landlord(info_set)
    elif info_set.player_position == 'landlord_up':
        return _get_obs_landlord_up(info_set)
    elif info_set.player_position == 'landlord_down':
        return _get_obs_landlord_down(info_set)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    卡牌梳理编码
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    将动作(即卡片矩阵中的整数列表, 在这里删除始终为零并使其变平的六个条目)
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

def _action_seq_list2array(action_seq_list):
    """
    对历史动作进行编码的实用函数。对历史上的15个动作进行编码。
    如果有没有15个操作，用0填充功能。自从三个动作在斗地主中是一个回合，连接每连续三次移动的表示。
    最后，得到一个5x162矩阵，它将被馈送转换为LSTM进行编码
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    一种对历史动作进行编码的实用函数。编码15次移动，如果没有15个动作，使用零
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    获取炸弹热编码
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(info_set):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(info_set.legal_actions)
    my_handcards = _cards2array(info_set.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],num_legal_actions, axis=0)

    other_handcards = _cards2array(info_set.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_up_num_cards_left = _get_one_hot_array(info_set.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(landlord_up_num_cards_left[np.newaxis, :],num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(info_set.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(landlord_down_num_cards_left[np.newaxis, :],num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(info_set.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(landlord_up_played_cards[np.newaxis, :],num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(info_set.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(landlord_down_played_cards[np.newaxis, :],num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        info_set.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    # 卡牌打出的动作序列
    z = _action_seq_list2array(_process_action_seq(
        info_set.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
    obs = {
        'position': 'landlord',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': info_set.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs

def _get_obs_landlord_up(info_set):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(info_set.legal_actions)
    my_handcards = _cards2array(info_set.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(info_set.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        info_set.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        info_set.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        info_set.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        info_set.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        info_set.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    # 玩家状态批次信息(存在重复数据)
    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    # 玩家状态只存在单条数据，无重复数据
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    # 玩家出牌序列编码，单条数据
    z = _action_seq_list2array(_process_action_seq(info_set.card_play_action_seq))
    # 重复数据
    z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
    # 玩家状态信息编码数据
    obs = {
        'position': 'landlord_up',
        'x_batch': x_batch.astype(np.float32),  # 玩家所有数据(存在重复数据)
        'z_batch': z_batch.astype(np.float32),  # 玩家出牌数据(存在重复数据)
        'legal_actions': info_set.legal_actions,  # 玩家合法动作
        'x_no_action': x_no_action.astype(np.int8),  # 玩家所有数据(只存在单挑数据)
        'z': z.astype(np.int8), # 玩家出牌数据(只存在单挑数据)
    }
    return obs

def _get_obs_landlord_down(info_set):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(info_set.legal_actions)
    my_handcards = _cards2array(info_set.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(info_set.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(info_set.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(info_set.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        info_set.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        info_set.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        info_set.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        info_set.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        info_set.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        info_set.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(info_set.bomb_num)
    bomb_num_batch = np.repeat(bomb_num[np.newaxis, :],num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    # np.hstack:参数元组的元素数组按水平方向进行叠加
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        info_set.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :],num_legal_actions, axis=0)
    obs = {
        'position': 'landlord_down',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': info_set.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs