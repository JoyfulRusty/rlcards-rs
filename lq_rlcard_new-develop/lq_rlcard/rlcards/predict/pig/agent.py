# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

# 基础路径
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pig", "models")
# 权重
WEIGHTS = 1129548800 # 1033779200  # 855500800
MODEL_PATH_1 = os.path.join(MODEL_PATH, f"landlord1_weights_{WEIGHTS}.ckpt")
MODEL_PATH_2 = os.path.join(MODEL_PATH, f"landlord2_weights_{WEIGHTS}.ckpt")
MODEL_PATH_3 = os.path.join(MODEL_PATH, f"landlord3_weights_{WEIGHTS}.ckpt")
MODEL_PATH_4 = os.path.join(MODEL_PATH, f"landlord4_weights_{WEIGHTS}.ckpt")


# TODO: 读取供猪训练模型
class DeepAgent:
    """
    封装拱猪预测模型
    """

    def __init__(self, position, model_path):
        """
        读取加载模型
        """
        self.model = self.load_model(position, model_path)

    def act(self, obs):
        """
        预测模型输出动作
        """
        if len(obs['legal_actions']) == 1:
            return obs['legal_actions'][0]

        # 转化参数
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()

        # 判断cuda是否可用
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()

        # 预测值
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        # 选取最好的idx
        best_action_index = np.argmax(y_pred, axis=0)[0]

        # 输出最佳动作
        best_action = obs['legal_actions'][best_action_index]

        return best_action

    @staticmethod
    def load_model(position, model_path):
        # from rlcards.games.pig.dmc.models import model_dict
        from .model_forward import model_dict
        model = model_dict[position]()
        model_state_dict = model.state_dict()
        if torch.cuda.is_available():
            pretrained = torch.load(model_path, map_location='cuda:0')
        else:
            pretrained = torch.load(model_path, map_location='cpu')
        pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model


choice_model = {
    'landlord1': DeepAgent('landlord1', MODEL_PATH_1),
    'landlord2': DeepAgent('landlord2', MODEL_PATH_2),
    'landlord3': DeepAgent('landlord3', MODEL_PATH_3),
    'landlord4': DeepAgent('landlord4', MODEL_PATH_4),
}

if __name__ == '__main__':
    print(choice_model)
