# -*- coding: utf-8 -*-

import os
import torch

# 模型路径
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "sytx_gz", "models")

# 模型权重
WEIGHTS = 155571200 # 294432000 # 155571200 # 69705600 # 112694400 # 504764800  # 47744000
MODEL_PATH_0 = os.path.join(BASE_PATH, f"0_{WEIGHTS}.pth")
MODEL_PATH_1 = os.path.join(BASE_PATH, f"1_{WEIGHTS}.pth")

# TODO: 预测模型读取
model_0 = torch.load(MODEL_PATH_0)
model_1 = torch.load(MODEL_PATH_1)

# 选择模型["庄家", "农民"]
choice_model = {"landlord": model_0, "farmer": model_1}

if __name__ == '__main__':
	print(model_0.state_dict())
	print(model_1.state_dict())