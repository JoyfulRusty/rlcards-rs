# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F

from torch import nn

from predict.ddz.xxc.utils import encode2onehot_by_real, encode2onehot


class BidModel1(nn.Module):
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
MODEL_PATH = os.path.join(BASE_PATH, f"bid_weights.pkl")

class BidModel:
	"""
	Bidirectional LSTM Model for predicting the next word in a sequence.
	"""

	def __init__(self, device, use_gpu=False):
		"""
		Initialize the Bidirectional LSTM Model.
		"""
		self.device = device
		self.use_gpu = use_gpu
		self.model = BidModel1()
		self.init_model()

	def init_model(self):
		"""
		Initialize the model weights.
		"""
		if self.use_gpu:
			self.model = self.model.to(self.device)
		if torch.cuda.is_available():
			self.model.load_state_dict(torch.load(MODEL_PATH))
		else:
			self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
		self.model.eval()

	def predict(self, cards):
		"""
		Predict the next word in a sequence.
		"""
		cards = encode2onehot_by_real(cards)
		if self.use_gpu:
			cards = cards.to(self.device)
		cards = torch.flatten(cards)
		win_rate = self.model(cards)
		return win_rate[0].item() * 100

	def predict_score(self, cards):
		"""
		Predict the next word in a sequence.
		"""
		cards = encode2onehot(cards)
		if self.use_gpu:
			cards = cards.to(self.device)
		cards = torch.flatten(cards)
		cards = cards.unsqueeze(0)
		res = self.model(cards)
		return res[0].item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predict_bid_agent = BidModel(device, use_gpu=True)

if __name__ == '__main__':
	model_res = predict_bid_agent.predict_score([18, 20, 6, 6, 6, 6, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12])
	print(model_res)