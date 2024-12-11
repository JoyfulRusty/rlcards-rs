from enum import IntEnum, unique

pokers_map = {
    3: [103, 203, 303, 403],
    4: [104, 204, 304, 404],
    5: [105, 205, 305, 405],
    6: [106, 206, 306, 406],
    7: [107, 207, 307, 407],
    8: [108, 208, 308, 408],
    9: [109, 209, 309, 409],
    10: [110, 210, 310, 410],
    11: [111, 211, 311, 411],
    12: [112, 212, 312, 412],
    13: [113, 213, 313, 413],
    14: [114, 214, 314, 414],
    16: [116, 216, 316, 416],
    18: [518, 520],
}


@unique
class CombType(IntEnum):
    """ 斗地主所有牌型组合 """
    ZHA_DAN = 1
    SHUN_ZI = 2
    LIAN_DUI = 3
    FEI_JI = 4
    DUI_ZI = 5

    @classmethod
    def all_value(cls):
        elements = []
        for v in cls._value2member_map_:
            elements.append(v)

        return elements


class CombProb:
    """ 选取组合概率 """
    prob = {
        CombType.ZHA_DAN: 5,
        CombType.SHUN_ZI: 50,
        CombType.LIAN_DUI: 30,
        CombType.FEI_JI: 15,
        CombType.DUI_ZI: 0,
    }


class TeShu(IntEnum):
    VAL1 = 10
    VAL2 = 7


class ShunNum(IntEnum):
    """ 单顺 """
    MIN_NUM = 5  # 最少张
    MAX_NUM = 12  # 最多张


class LianDuiNum(IntEnum):
    MIN_NUM = 3
    MAX_NUM = 8  # 17 // 2


class FeiJiNum(IntEnum):
    MIN_NUM = 1
    MAX_NUM = 5


class LianZhaNum(IntEnum):
    MIN_NUM = 1
    MAX_NUM = 4
