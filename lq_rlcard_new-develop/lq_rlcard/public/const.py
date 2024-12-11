import os
from enum import IntEnum, unique, Enum

LOG_DEBUG = False

# 项目根目录
BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 输出目录路径
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# 配置路径
CONF_PATH = os.path.join(BASE_PATH, "config", "conf.json")


@unique
class ServerType(IntEnum):
    """ 服务类型 """
    SERVICE_DYG_XXC = 76  # 打妖怪休闲场


@unique
class HandleType(IntEnum):
    """ 处理函数类型 """
    TEST = 100


@unique
class ChannelType(str, Enum):
    """ redis频道 """
    CHANNEL_GATEWAY_CHILD = "channel_gateway_child"  # 子游戏网关
