# -*- coding: utf-8 -*-

import os

from predict.monster.deep_agent import DeepAgent

# 模型路径
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "monster", "model")
NEW_WEIGHTS = 65808000
DOWN_MODEL_PATH = os.path.join(BASE_PATH, f"down_{NEW_WEIGHTS}.pth")
RIGHT_MODEL_PATH = os.path.join(BASE_PATH, f"right_{NEW_WEIGHTS}.pth")
UP_MODEL_PATH = os.path.join(BASE_PATH, f"up_{NEW_WEIGHTS}.pth")
LEFT_MODEL_PATH = os.path.join(BASE_PATH, f"left_{NEW_WEIGHTS}.pth")

DYG_MODEL_PATHS = {
	"down": DOWN_MODEL_PATH,
	"right": RIGHT_MODEL_PATH,
	"up": UP_MODEL_PATH,
	"left": LEFT_MODEL_PATH
}

dyg_model_dict = {
	position: DeepAgent(position=position, model_path=model_path)for position, model_path in DYG_MODEL_PATHS.items()
}