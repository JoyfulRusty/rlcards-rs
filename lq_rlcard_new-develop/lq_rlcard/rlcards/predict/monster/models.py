# -*- coding: utf-8 -*-

import os
import torch

# 模型路径
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "monster", "models")

# 模型权重
WEIGHTS = 235292800  # 15680000  # 235292800  # 258563200
MODEL_PATH_1 = os.path.join(BASE_PATH, f"0_{WEIGHTS}.pth")
MODEL_PATH_2 = os.path.join(BASE_PATH, f"1_{WEIGHTS}.pth")
MODEL_PATH_3 = os.path.join(BASE_PATH, f"2_{WEIGHTS}.pth")
MODEL_PATH_4 = os.path.join(BASE_PATH, f"3_{WEIGHTS}.pth")

# TODO: 读取预测模型
model_0 = torch.load(MODEL_PATH_1)
model_1 = torch.load(MODEL_PATH_2)
model_2 = torch.load(MODEL_PATH_3)
model_3 = torch.load(MODEL_PATH_4)

# 选择模型
choice_model = {
	0: model_0,
	1: model_1,
	2: model_2,
	3: model_3
}

if __name__ == '__main__':
	print(model_0)
	print(model_1)
	print(model_2)
	print(model_3)