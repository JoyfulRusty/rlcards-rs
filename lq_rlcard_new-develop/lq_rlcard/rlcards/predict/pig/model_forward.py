"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(208, 128, batch_first=True)  # batch_first ，表示输入的 Tensor 的第一个维度是batch 信息
        self.dense1 = nn.Linear(884 + 128, 1024)  # x_batch + z_batch -> lstm -> 128
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 256)
        self.dense5 = nn.Linear(256, 128)
        self.dense6 = nn.Linear(128, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        # print("x1: ", x, "x1_shape: ", x.shape)
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                # print("x2: ", x, "x2_shape: ", x.shape)
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


# Model dict is only used in evaluation but not training
model_dict = {
    'landlord1': LandlordLstmModel,
    'landlord2': LandlordLstmModel,
    'landlord3': LandlordLstmModel,
    'landlord4': LandlordLstmModel,
}


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """

    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord1'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord2'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord3'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord4'] = LandlordLstmModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord1'].share_memory()
        self.models['landlord2'].share_memory()
        self.models['landlord3'].share_memory()
        self.models['landlord4'].share_memory()

    def eval(self):
        self.models['landlord1'].eval()
        self.models['landlord2'].eval()
        self.models['landlord3'].eval()
        self.models['landlord4'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
