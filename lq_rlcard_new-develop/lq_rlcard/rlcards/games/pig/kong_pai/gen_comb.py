import random
from rlcards.games.pig.kong_pai.poker_const import TeShu, LianDuiNum, FeiJiNum, ShunNum


class GenComb:
    @staticmethod
    def get_zha_dan(cards: dict, num_len=1):
        all_bomb = []
        # if not cards.get(TeShu.VAL1):
        #     cards.pop(TeShu.VAL2, 1)
        # if not cards.get(TeShu.VAL2):
        #     cards.pop(TeShu.VAL1, 1)
        for k, v in cards.items():
            if len(v) == 4:
                all_bomb.append(k)
        if num_len == 1:
            return all_bomb and [random.sample(all_bomb, 1)]
        return GenComb._gen_serial_moves(all_bomb, num_len)

    @staticmethod
    def get_fei_ji(cards: dict, num_len=1, repeat_num_max=FeiJiNum.MAX_NUM):
        all_fj = []
        # if not cards.get(TeShu.VAL1):
        #     cards.pop(TeShu.VAL2, 1)
        # if not cards.get(TeShu.VAL2):
        #     cards.pop(TeShu.VAL1, 1)
        for k, v in cards.items():
            if len(v) >= 3:
                all_fj.append(k)
        if num_len == 1:
            return all_fj and [random.sample(all_fj, 1)]
        return GenComb._gen_serial_moves(all_fj, num_len, repeat_num_max)

    @staticmethod
    def get_lian_dui(cards: dict, num_len=3, repeat_num_max=LianDuiNum.MAX_NUM):
        all_pairs = []
        for k, v in cards.items():
            if len(v) >= 2:
                all_pairs.append(k)
        if num_len == 1:
            return all_pairs and [random.sample(all_pairs, 1)]
        return GenComb._gen_serial_moves(all_pairs, num_len, repeat_num_max)

    @staticmethod
    def get_pairs(cards: dict, num_len=1):
        return GenComb.get_lian_dui(cards, num_len)

    @staticmethod
    def get_shun_zi(cards, num_len=5, repeat_num_max=ShunNum.MAX_NUM):
        if isinstance(cards, dict):
            cards = list(cards.keys())
        return GenComb._gen_serial_moves(cards, num_len, repeat_num_max)

    @staticmethod
    def _gen_serial_moves_dan(cards, num):
        """
        将cards split一个个连续的 []
        """
        cursor = 0
        cards.sort()
        all_comb = []
        for i in range(len(cards)):
            if i == len(cards) - 1:
                ser_cards = cards[cursor:]
                if len(ser_cards) >= num:
                    all_comb.append(ser_cards)
                return all_comb
            if cards[i] + 1 != cards[i + 1]:
                ser_cards = cards[cursor:i + 1]
                if len(ser_cards) >= num:
                    all_comb.append(ser_cards)
                cursor = i + 1
        return all_comb

    @staticmethod
    def _gen_serial_moves(cards, min_serial=5, repeat_num_max=0, repeat=1):
        """
        cards: 手牌
        min_serial: 最小连数
        repeat: 每张牌的重复数
        repeat_num：最大重复牌长度
        """
        if repeat_num_max < min_serial:  # at least repeat_num is min_serial
            repeat_num_max = 0

        single_cards = sorted(list(set(cards)))
        seq_records = list()
        moves = list()

        start = i = 0
        longest = 1
        while i < len(single_cards):
            if i + 1 < len(single_cards) and single_cards[i + 1] - single_cards[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        for seq in seq_records:
            if seq[1] < min_serial:
                continue
            start, longest = seq[0], seq[1]
            longest_list = single_cards[start: start + longest]

            if repeat_num_max == 0:
                steps = min_serial
                while steps <= longest:
                    index = 0
                    while steps + index <= longest:
                        end = steps + index
                        target_moves = sorted(longest_list[index: end] * repeat)
                        moves.append(target_moves)
                        index += 1
                    steps += 1
            else:
                # if longest < min_serial:
                #     continue
                steps = min_serial
                while steps <= longest:
                    if steps > repeat_num_max:
                        break
                    index = 0
                    while steps + index <= longest:
                        target_moves = sorted(longest_list[index: steps + index])
                        moves.append(target_moves)
                        index += 1
                    steps += 1
        return moves


if __name__ == '__main__':
    a = [1, 2, 4, 5, 6]
    res = GenComb._gen_serial_moves(a, 1)
    print(res)
