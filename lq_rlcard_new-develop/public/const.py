# -*- coding: utf-8 -*-

import os

LOG_DEBUG = False

# 项目根目录
BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 输出目录路径
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# 配置路径
CONF_PATH = os.path.join(BASE_PATH, "config", "conf.json")

DEFAULT_CONFIG = {"allow_step_back": False, "seed": None}  # default config

ENV_ENTRY_POINTS = {
	"monster": {"env_id": "monster", "entry_point": "reinforce.env.monster:MonsterEnv"},
}