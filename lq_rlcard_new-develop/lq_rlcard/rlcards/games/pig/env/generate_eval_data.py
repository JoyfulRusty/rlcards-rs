import argparse
import pickle
import numpy as np

from rlcards.const.pig.const import ALL_POKER

deck = ALL_POKER[:]


def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=10000, type=int)
    return parser


def generate():
    _deck = deck.copy()
    _deck.remove(407)
    np.random.shuffle(_deck)
    card_play_data = {'landlord1': [407] + _deck[:12],
                      'landlord2': _deck[12:25],
                      'landlord3': _deck[25:38],
                      'landlord4': _deck[38:51],
                      }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'

    print("output_pickle:", output_pickle)
    print("generating data...")

    data = []
    for _ in range(flags.num_games):
        data.append(generate())

    print("saving pickle file...")
    with open(output_pickle, 'wb') as g:
        pickle.dump(data, g, pickle.HIGHEST_PROTOCOL)
