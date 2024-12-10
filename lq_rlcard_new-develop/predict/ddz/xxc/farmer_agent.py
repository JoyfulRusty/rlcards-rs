# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F

from torch import nn

from predict.ddz.xxc.utils import encode2onehot


class FarmerModel1(nn.Module):
    """
    Bidirectional LSTM Model for predicting the next word in a sequence.
    """

    def __init__(self):
        """
        Initialize the Bidirectional LSTM Model.
        """
        super().__init__()
        # input: 1 * 60
        self.conv1 = nn.Conv1d(1, 16, kernel_size=(3,), padding=1)  # 32 * 60
        self.dense1 = nn.Linear(1020, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 1)

    def forward(self, xi):
        x = xi.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = x.flatten(1, 2)
        x = torch.cat((x, xi), 1)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = F.leaky_relu(self.dense4(x))
        x = self.dense5(x)
        return x


BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pkl")
UP_MODEL_PATH = os.path.join(BASE_PATH, f"score_landlord_up.pkl")
DOWN_MODEL_PATH = os.path.join(BASE_PATH, f"score_landlord_down.pkl")
FARMER_MODEL_PATH = os.path.join(BASE_PATH, f"score_farmer.pkl")


class FarmerModel:
    """
    Bidirectional LSTM Model for predicting the next word in a sequence.
    """

    def __init__(self, device, use_gpu=False):
        """
        Initialize the Bidirectional LSTM Model.
        """
        self.device = device
        self.use_gpu = use_gpu
        self.model = {
            "up": FarmerModel1(),
            "down": FarmerModel1(),
            "farmer": FarmerModel1(),
        }

        # 初始化加载模型
        self.init_model()

    def init_model(self):
        """
        Initialize the model weights.
        """
        if torch.cuda.is_available():
            self.model["up"].load_state_dict(torch.load(UP_MODEL_PATH, map_location=self.device))
            self.model["down"].load_state_dict(torch.load(DOWN_MODEL_PATH, map_location=self.device))
            self.model["farmer"].load_state_dict(torch.load(FARMER_MODEL_PATH, map_location=self.device))
        else:
            self.model["up"].load_state_dict(torch.load(UP_MODEL_PATH, map_location=torch.device('cpu')))
            self.model["down"].load_state_dict(torch.load(DOWN_MODEL_PATH, map_location=torch.device('cpu')))
            self.model["farmer"].load_state_dict(torch.load(FARMER_MODEL_PATH, map_location=torch.device('cpu')))

        # eval
        self.model["up"].eval()
        self.model["down"].eval()
        self.model["farmer"].eval()

    def predict(self, cards, model_ps="up"):
        """
        Predict the next word in a sequence.
        """
        model = self.model[model_ps]
        cards = encode2onehot(cards)
        x = torch.flatten(cards)
        x = x.unsqueeze(0)
        y = model(x)
        return y.squeeze().item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predict_farmer_agent = FarmerModel(device, use_gpu=True)

if __name__ == '__main__':
    model_res = predict_farmer_agent.predict([18, 20, 6, 6, 6, 6, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12])
    print(model_res)
