"""
亮牌策略
"""
CAN_LIGHT_CARDS = [412, 111, 210, 316]
BAN = 210  # 变压器
YANG = 111  # 羊牌
ZHU = 412  # 猪
KING = 316  # 大鬼
HX_J = 311  # 红桃J


def lp_policy(cards, yet_lp):
    """ 亮牌策略 """
    light_cards = []
    can_lp = list(set(cards).intersection(CAN_LIGHT_CARDS))
    if not can_lp:
        return light_cards
    if not cards or not isinstance(cards, list):
        return light_cards
    suit_cards = dict()
    for c in cards:
        suit = int(c / 100)
        suit_cards.setdefault(suit, []).append(c)

    cards_dis = get_cards_distribution(suit_cards)

    for cl in can_lp:
        suit = int(cl / 100)
        type_cards = suit_cards.get(suit)
        if cl == ZHU:
            if len(type_cards) >= 5:
                light_cards.append(cl)
        elif cl == YANG:
            if len(type_cards) >= 4:
                if cards_dis != [3, 3, 3, 4]:
                    light_cards.append(cl)
        elif cl == BAN:
            if ZHU not in yet_lp and YANG not in yet_lp:
                if len(type_cards) >= 5:
                    light_cards.append(cl)
        elif cl == KING:
            if len(type_cards) >= 5:
                if len(gt_or_lt_xx_cards(type_cards, HX_J)) >= 5:
                    light_cards.append(cl)
                elif 315 not in type_cards:
                    if cal_xx_cards_num(cards_dis) > 0:
                        light_cards.append(cl)

    return light_cards


def get_cards_distribution(suit_cards):
    """ 根据牌花色数量排序 """
    cards_dis = []
    if not suit_cards:
        return cards_dis
    for c in suit_cards:
        cards_dis.append(len(suit_cards.get(c, [])))
    cards_dis.sort()
    return cards_dis


def gt_or_lt_xx_cards(cards, xx, lt=False):
    """ 大于或者小于xx牌 """
    gt_cards = []
    for c in cards:
        if not lt and c >= xx:
            gt_cards.append(c)
        if lt and c <= xx:
            gt_cards.append(c)
    return gt_cards


def cal_xx_cards_num(cards, num=2):
    """ 分区牌数小于2的 """
    count = 0
    for c in cards:
        if c <= num:
            count += 1
    return count


if __name__ == '__main__':
    hand = [107, 110, 114, 203, 305, 306, 309, 313, 316, 406, 407, 410, 412]
    res = lp_policy(hand, [])
    print(res)