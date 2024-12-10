# -*- coding: utf-8 -*-

import os

from predict.ddz.xxc.deep_agent import DeepAgent
from predict.ddz.xxc.const import ALL_POSITION, ALL_POSITION_4_AN

# 模型路径
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "ddz", "model")

ADP_LANDLORD_UP_PATH = os.path.join(BASE_PATH, f"adp_landlord_up.ckpt")
ADP_LANDLORD_PATH = os.path.join(BASE_PATH, f"adp_landlord.ckpt")
ADP_LANDLORD_DOWN_PATH = os.path.join(BASE_PATH, f"adp_landlord_down.ckpt")
ADP_MODEL_PATHS = {
	"landlord_up": ADP_LANDLORD_UP_PATH,
	"landlord": ADP_LANDLORD_PATH,
	"landlord_down": ADP_LANDLORD_DOWN_PATH,
}

AN_LANDLORD1_PATH = os.path.join(BASE_PATH, f"an_landlord1.ckpt")
AN_LANDLORD2_PATH = os.path.join(BASE_PATH, f"an_landlord2.ckpt")
AN_LANDLORD3_PATH = os.path.join(BASE_PATH, f"an_landlord3.ckpt")
AN_LANDLORD4_PATH = os.path.join(BASE_PATH, f"an_landlord4.ckpt")
AN_MODEL_PATHS = {
	"landlord1": AN_LANDLORD1_PATH,
	"landlord2": AN_LANDLORD2_PATH,
	"landlord3": AN_LANDLORD3_PATH,
	"landlord4": AN_LANDLORD4_PATH,
}

# 不洗牌
BXP_LANDLORD_UP_PATH = os.path.join(BASE_PATH, f"bxp_landlord_up.ckpt")
BXP_LANDLORD_PATH = os.path.join(BASE_PATH, f"bxp_landlord.ckpt")
BXP_LANDLORD_DOWN_PATH = os.path.join(BASE_PATH, f"bxp_landlord_down.ckpt")
BXP_MODEL_PATHS = {
	"landlord_up": BXP_LANDLORD_UP_PATH,
	"landlord": BXP_LANDLORD_PATH,
	"landlord_down": BXP_LANDLORD_DOWN_PATH,
}

SELF_ADP_LANDLORD_UP_PATH = os.path.join(BASE_PATH, f"self_adp_landlord_up.ckpt")
SELF_ADP_LANDLORD_PATH = os.path.join(BASE_PATH, f"self_adp_landlord.ckpt")
SELF_ADP_LANDLORD_DOWN_PATH = os.path.join(BASE_PATH, f"self_adp_landlord_down.ckpt")
SELF_ADP_MODEL_PATHS = {
	"landlord_up": SELF_ADP_LANDLORD_UP_PATH,
	"landlord": SELF_ADP_LANDLORD_PATH,
	"landlord_down": SELF_ADP_LANDLORD_DOWN_PATH,
}

WP_LANDLORD_PATH = os.path.join(BASE_PATH, f"wp_landlord.ckpt")


# 加载构建ADP MODEL
ADP_MODELS = {
	position: DeepAgent(position, ADP_MODEL_PATHS.get(position)) for position in ALL_POSITION
}

# 加载构建AN MODEL
AN_MODELS = {
	position: DeepAgent(position, AN_MODEL_PATHS.get(position), is_4_an=True) for position in ALL_POSITION_4_AN
}

# 加载构建BXP MODEL
BXP_MODELS = {
	position: DeepAgent(position, BXP_MODEL_PATHS.get(position)) for position in ALL_POSITION
}

# 加载构建SELF_ADP MODEL
SELF_ADP_MODES = {
	position: DeepAgent(position, SELF_ADP_MODEL_PATHS.get(position)) for position in ALL_POSITION
}

# 加载构建WP MODEL
WP_MODELS = {
	"landlord": DeepAgent("landlord", WP_LANDLORD_PATH)
}


MODEL_MAPS = {
	0: "landlord_up",
	1: "landlord",
	2: "landlord_down",
}

AN_MODEL_MAPS = {
	0: "landlord1",
	1: "landlord2",
	2: "landlord3",
	3: "landlord4",
}

DEALER_MODEL_MAPS = {
	1: {0: "landlord", 1: "landlord_down", 2: "landlord_up"},
	2: {0: "landlord_up", 1: "landlord", 2: "landlord_down"},
	3: {0: "landlord_down", 1: "landlord_up", 2: "landlord"},
}