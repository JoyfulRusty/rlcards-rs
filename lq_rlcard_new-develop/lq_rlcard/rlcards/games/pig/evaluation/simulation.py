import multiprocessing as mp
import pickle

from rlcards.games.pig.env.game import GameEnv
from rlcards.games.pig.env.utils import ALL_ROLE


def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ALL_ROLE:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players


def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    players = load_card_play_models(card_play_model_path_dict)

    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()

    q.put((
        env.num_wins['landlord1'],
        env.num_wins['landlord2'],
        env.num_wins['landlord3'],
        env.num_wins['landlord4'],
        env.num_scores['landlord1'],
        env.num_scores['landlord2'],
        env.num_scores['landlord3'],
        env.num_scores['landlord4'],
    ))


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker


def evaluate(landlord1, landlord2, landlord3, landlord4, eval_data, num_workers):
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord1': landlord1,
        'landlord2': landlord2,
        'landlord3': landlord3,
        'landlord4': landlord4,
    }

    num_landlord1_wins = 0
    num_landlord2_wins = 0
    num_landlord3_wins = 0
    num_landlord4_wins = 0
    num_landlord1_scores = 0
    num_landlord2_scores = 0
    num_landlord3_scores = 0
    num_landlord4_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
            target=mp_simulate,
            args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord1_wins += result[0]
        num_landlord2_wins += result[1]
        num_landlord3_wins += result[2]
        num_landlord4_wins += result[3]
        num_landlord1_scores += result[4]
        num_landlord2_scores += result[5]
        num_landlord3_scores += result[6]
        num_landlord4_scores += result[7]

    num_total_wins = num_landlord1_wins + num_landlord2_wins + num_landlord3_wins + num_landlord4_wins
    print("总赢：", num_total_wins)
    scale = (
        num_landlord1_wins / num_total_wins, num_landlord2_wins / num_total_wins, num_landlord3_wins / num_total_wins,
        num_landlord4_wins / num_total_wins)
    scale_score = (
        num_landlord1_scores / num_total_wins, num_landlord2_scores / num_total_wins,
        num_landlord3_scores / num_total_wins,
        num_landlord4_scores / num_total_wins)
    print('WP results:')
    print('landlord1 : landlord2 : landlord3 : landlord4 - {} : {} : {} : {}'.format(*scale))
    print('ADP results:')
    print('landlord1 : landlord2 : landlord3 : landlord4 - {} : {} : {} : {}'.format(*scale_score))
    print("四个赢的次数", num_landlord1_wins, num_landlord2_wins, num_landlord3_wins, num_landlord4_wins)
    print("四个赢的分数", num_landlord1_scores, num_landlord2_scores, num_landlord3_scores, num_landlord4_scores)
    print("total_wins: ", num_total_wins)
