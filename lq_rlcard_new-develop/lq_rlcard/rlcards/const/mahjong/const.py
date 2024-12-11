# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum, IntEnum, unique


# TODO: 常量设置
LAI_ZI = 80  # 癞子牌
PLAYERS_NUM = 3  # 玩家人数
INIT_REWARD = [0.0, 0.0, 0.0]  # 初始奖励

BIRD_VALUE = (1, 5, 9)  # 鸡牌
MAX_PICK_NUMS = 18  # 牌桌中卡牌数量小于3张不能捡了
XUE_LIU_LEFT_BI_HU = 18  # 血流必胡最小牌数

ACTION_TYPE_MING_GANG = 2  # 明杠
ACTION_TYPE_ZHUAN_WAN_GANG = 3  # 转弯杠
ACTION_TYPE_AN_GANG = 4  # 暗杠

# TODO: 卡牌编码信息(动作对应的索引)
Card2Column = {
    80: 27,
    11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8,
    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17,
    31: 18, 32: 19, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26,
    'PASS': 28, 'TIAN_TING': 29, 'PONG': 30, 'MING_GONG': 31, 'SUO_GONG': 32,
    'AN_GONG': 33, 'GS_PAO': 34, 'QG_HU': 35, 'PICK_HU': 36, 'MI_HU': 37,
    'KAI_HU': 38
}

# TODO: 编码卡牌出现的次数
NumOnes2Array = {
    0: np.array([0, 0, 0, 0]),  # 0次
    1: np.array([1, 0, 0, 0]),  # 1次
    2: np.array([1, 1, 0, 0]),  # 2次
    3: np.array([1, 1, 1, 0]),  # 3次
    4: np.array([1, 1, 1, 1])   # 4次
}

# 卡牌类型编码信息(万、条、筒、癞子)
CardType2Column = {
    1: 0,  # 万
    2: 1,  # 条
    3: 2,  # 筒
    8: 3   # 癞
}

# 卡牌编码信息
NumTypeOnes2Array = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    1: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    2: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    3: np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    4: np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    5: np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    6: np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    7: np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
    8: np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    9: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    10: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
    11: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
    12: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    13: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
    14: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
}

# 卡牌(万条筒)
ALL_CARDS = (
    11, 12, 13, 14, 15, 16, 17, 18, 19,  # 万字: 11~19
    21, 22, 23, 24, 25, 26, 27, 28, 29,  # 线条: 21~29
    31, 32, 33, 34, 35, 36, 37, 38, 39,  # 筒子: 31~39
)

CARD_TYPES = {
    1: 'chars',
    2: 'bamboos',
    3: 'dots',
    8: 'lai_zi'
}

CARD_VALUE = [
    '80',
    '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '31', '32', '33', '34', '35', '36', '37', '38', '39',
]

# 卡牌字符索引
CARD2StrIdx = {
    0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18, 8: 19,
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29,
    18: 31, 19: 32, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39
}


# TODO: 卡牌相关类型(结束类型、位置类型、卡牌类型等)
@unique
class OverType(IntEnum):
    """ 游戏结束类型 """
    DEFAULT = -1  # 默认值
    LIU_JU = 1  # 流局，游戏结束无赢家
    HU_KAI = 2  # 胡开，游戏结束有赢家


@unique
class SuitType(IntEnum):
    """
    花色类型
    """
    SUIT_WAN = 1  # 万
    SUIT_SUO = 2  # 条
    SUIT_TONG = 3  # 筒
    SUIT_HONG = 8  # 红中


class CardType(IntEnum):
    """
    卡牌类型
    """
    YAO_JI = 21
    WU_GU_JI = 38
    LAI_ZI = 51


# 默认鸡牌(幺鸡和乌骨鸡)
DEFAULT_JI_CARDS = {
    CardType.YAO_JI,
    CardType.WU_GU_JI
}


@unique
class CombType(IntEnum):
    """
    卡牌连续类型
    """
    TYPE_KE_ZI = 1  # 刻子
    TYPE_SHUN_ZI = 2  # 顺子
    TYPE_SI_GANG = 3  # 死杠
    TYPE_DUI_ZI = 4  # 大对
    TYPE_SHUN_ZI_2 = 5  # 二连顺
    TYPE_GAP_SHUN = 6  # 间隔顺
    TYPE_SINGLE = 7  # 单张


@unique
class ActionType(str, Enum):
    """
    红中捡漏动作类型
    """
    PONG = "PONG"  # 碰
    TIAN_TING = "TIAN_TING"  # 报听
    MING_GONG = "MING_GONG"  # 明杠
    AN_GONG = "AN_GONG"  # 暗杠
    SUO_GONG = "SUO_GONG"  # 索杠/转弯杠
    PICK_HU = "PICK_HU"  # 捡胡
    MI_HU = "MI_HU"  # 密胡
    KAI_HU = "KAI_HU"  # 开胡(游戏结束)
    G_PASS = "PASS"  # 上一位打牌，其余玩家都不要
    GS_PAO = "GS_PAO"  # 杠上炮
    QG_HU = "QG_HU"  # 抢杠胡


@unique
class HuPaiType(IntEnum):
    """ 红中捡漏胡牌类型 """
    # 基础牌型
    PING_HU = 101  # 平胡
    DA_DUI_ZI = 102  # 大对子(4副刻子(三张相同的牌)加上1个对子)
    QI_DUI = 103  # 七对(手中由7副对子组成的牌型)
    LONG_QI_DUI = 104  # 龙七对(5个对子+1刻子)豪华七对
    DI_LONG_QI = 105  # 地龙七(5个对子+4张相同的牌)
    QING_YI_SE = 106  # 清一色
    QING_DA_DUI = 107  # 清大对(清一色的大对子)
    QING_QI_DUI = 108  # 清七对(清一色的小七对)
    QING_DI_LONG = 109  # 清地龙(清一色的地龙七)
    QING_LONG_QI = 110  # 清龙对(清一色的龙七对/清龙背/清龙七对)


@unique
class ExtraHuPaiType(IntEnum):
    """
    额外胡牌类型定义
    """
    # 额外胡牌类型，在基础胡牌类型之上，但与牌型无关
    TIAN_HU = 201  # 天胡(若为平胡，则不算再算平胡分)
    DI_HU = 202  # 地胡
    TIAN_TING = 203  # 天听
    SHA_BAO = 204  # 杀报
    GS_HUA = 205  # 杠上花(杠开)
    GS_PAO = 206  # 杠上炮(杠上炮(热炮))
    QG_HU = 207  # 抢杠胡 抢杠(被抢杠者根据抢杠的牌型分代替其他玩家给分)

    # 特殊胡牌型
    BAO_TING_QING_QI_DUI = 208  # 报听清七对
    BAO_TING_LONG_QI_DUI = 209  # 报听清龙七对


# 用于点炮情况下算分
SPECIAL_HU_TYPE = (ExtraHuPaiType.TIAN_TING, ExtraHuPaiType.SHA_BAO, ExtraHuPaiType.TIAN_HU)

# TODO: 胡牌类型和动作类型值
# 动作类型值
ACTION_PRIORITY = {
    ActionType.KAI_HU: 10,
    ActionType.TIAN_TING: 9,
    ActionType.QG_HU: 8,
    ActionType.MI_HU: 7,
    ActionType.PICK_HU: 6,
    ActionType.AN_GONG: 5,
    ActionType.SUO_GONG: 4,
    ActionType.MING_GONG: 3,
    ActionType.PONG: 2,
    ActionType.G_PASS: 1,
}

# 胡牌类型分数
HU_PAI_SCORES = {
    101: 1,
    102: 5,
    103: 10,
    104: 20,
    105: 20,
    106: 10,
    107: 15,
    108: 20,
    109: 30,
    110: 20,
    201: 40,
    202: 35,
    203: 30,
    204: 15,
    205: 45,
    206: 40,
    207: 35
}

# 额外分，和牌型无关，与时机有关
ACTION_SCORES_MAP = {
    ExtraHuPaiType.TIAN_HU: 20,
    ExtraHuPaiType.DI_HU: 2,
    ExtraHuPaiType.SHA_BAO: 10,
    ExtraHuPaiType.GS_HUA: 6,
    ExtraHuPaiType.GS_PAO: 1,
    ActionType.QG_HU: 2,
    ActionType.SUO_GONG: 3,
    ActionType.AN_GONG: 3,
    ActionType.TIAN_TING: 10,
    ActionType.MING_GONG: 3,
}


@unique
class JiType_XL(IntEnum):
    """
    结算鸡分类型
    """
    YAO_JI = 31  # 幺鸡
    FAN_PAI_JI = 32  # 翻牌鸡
    CF_JI = 33  # 冲锋鸡
    ZE_REN_JI = 34  # 责任鸡
    WU_GU_JI = 38  # 乌骨鸡(8筒，作用与幺鸡一样，分数是幺鸡的2倍)
    WU_GU_CFJ = 36  # 乌骨冲锋鸡，算6分
    WU_GU_ZRJ = 37  # 乌骨责任鸡，第一张打出的8筒被破碰杠，责任人赔付6分
    BAO_JI = 35  # 包鸡
    HAND_IN_JI = 39  # 手上鸡
    WU_GU_CF_JJI = 40  # 乌骨冲锋捡鸡


JI_PAI_SCORES_XL = {
    # 鸡牌分
    CardType.YAO_JI: 1,
    CardType.WU_GU_JI: 2,
    # 鸡牌类型分
    JiType_XL.YAO_JI: 1,
    JiType_XL.FAN_PAI_JI: 1,
    JiType_XL.CF_JI: 3,
    JiType_XL.ZE_REN_JI: 3,
    JiType_XL.WU_GU_JI: 2,
    JiType_XL.WU_GU_ZRJ: 6,
    JiType_XL.WU_GU_CFJ: 6
}


class JiType_FK(IntEnum):
    """
    鸡/杠type
    以2开头
    """
    DEFAULT = 201
    CF_JI = 202
    ZE_REN_JI = 203
    MAN_TANG_JI = 204
    SHANG_XIA_JI = 205
    BEN_JI = 206
    WU_GU_JI = 207
    WU_GU_CFJ = 208
    WU_GU_ZRJ = 209
    JIN_JI = 210
    WU_GU_JIN_JI = 211
    CF_JIN_JI = 212
    WU_GU_CF_JIN_JI = 213

    AN_GANG = 214  # 暗杠
    MING_GANG = 215  # 明杠
    SUO_GANG = 216  # 转弯杠


# 鸡牌分
JI_PAI_SCORE_FK = {
    # 鸡牌分(默认鸡)
    CardType.YAO_JI: 1,
    CardType.WU_GU_JI: 2,

    # 鸡牌类型分
    JiType_FK.DEFAULT: 1,
    JiType_FK.CF_JI: 3,
    JiType_FK.CF_JIN_JI: 6,
    JiType_FK.ZE_REN_JI: 3,
    JiType_FK.JIN_JI: 2,
    JiType_FK.WU_GU_JI: 2,
    JiType_FK.WU_GU_CFJ: 4,
    JiType_FK.WU_GU_ZRJ: 2,
    JiType_FK.WU_GU_JIN_JI: 4,
    JiType_FK.WU_GU_CF_JIN_JI: 8,

    # 此处给3分，其余外面逻辑判断
    JiType_FK.AN_GANG: 3,
    JiType_FK.MING_GANG: 3,
    JiType_FK.SUO_GANG: 3,
}

if __name__ == '__main__':
    print('输出卡牌信息: ', NumTypeOnes2Array)
    print()
    print('输出胡牌分数计算: ', HU_PAI_SCORES)
    print()