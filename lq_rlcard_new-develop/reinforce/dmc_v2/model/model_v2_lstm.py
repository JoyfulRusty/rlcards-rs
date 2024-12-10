# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch import nn


class LstmModel(nn.Module):
	"""
	LSTM model for the DMC_v2 environment.
	"""

	def __init__(self, mlp_layers=None):
		"""
		Initialize the LSTM model.
		"""
		super().__init__()
		if not mlp_layers:
			mlp_layers = [512, 512, 512, 512, 512]
		self.lstm = nn.LSTM(33, 17, batch_first=True)
		self.dense0 = nn.Linear(435, mlp_layers[0])
		self.dense1 = nn.Linear(mlp_layers[0], mlp_layers[1])
		self.dense2 = nn.Linear(mlp_layers[1], mlp_layers[2])
		self.dense3 = nn.Linear(mlp_layers[2], mlp_layers[3])
		self.dense4 = nn.Linear(mlp_layers[3], mlp_layers[4])
		self.dense5 = nn.Linear(mlp_layers[4], 1)

	def forward(self, z, x):
		"""
		Forward the model.
		"""
		lstm_z, (h_n, _) = self.lstm(z)
		lstm_z = lstm_z[:, -1, :]
		val_x = torch.cat([lstm_z, x], dim=-1)
		val_x = self.dense0(val_x)
		val_x = torch.relu(val_x)
		val_x = self.dense1(val_x)
		val_x = torch.relu(val_x)
		val_x = self.dense2(val_x)
		val_x = torch.relu(val_x)
		val_x = self.dense3(val_x)
		val_x = torch.relu(val_x)
		val_x = self.dense4(val_x)
		val_x = torch.relu(val_x)
		val_x = self.dense5(val_x)
		return val_x


class Model:
	"""
	Model for DMC v2.
	"""

	def __init__(self, positions, mlp_layers=None, exp_epsilon=0.1, device="0"):
		"""
		初始化参数
		"""
		self.positions = positions
		self.mlp_layers = mlp_layers
		self.exp_epsilon = exp_epsilon
		self.device = 'cuda:' + str(device) if device != "cpu" else "cpu"
		self.net = LstmModel(mlp_layers=self.mlp_layers).to(self.device)
		self.models = {position: self.net for position in self.positions}

	def predict(self, position, z, x, obs):
		"""
		Predict action function for DMC v2.
		"""
		legal_actions = obs["legal_actions"]
		# Only one legal action, return it
		if len(legal_actions) == 1:
			return legal_actions[-1]
		# Otherwise use the model to predict action
		model = self.models[position]
		output = self.forward(model, z=z, x=x)
		output = output.cpu().detach().numpy()
		action_idx = np.argmax(output, axis=0)[0]
		best_action = legal_actions[action_idx]
		return best_action

	@staticmethod
	def forward(model, z, x):
		"""
		Forward function for DMC v2.
		"""
		return model.forward(z=z, x=x)

	def parameters(self, position):
		"""
		Get the parameters for the model at a given position.
		"""
		return self.models[position].parameters()

	def get_model(self, position):
		"""
		Get the model at a given position.
		"""
		return self.models[position]

	def get_models(self):
		"""
		Get all the models.
		"""
		return self.models

	def load_state_dict(self, state_dict):
		"""
		Load the state dict from the model
		"""
		return self.net.load_state_dict(state_dict)

	def state_dict(self):
		"""
		Get the state dict from the model
		"""
		return self.net.state_dict()

	def share_memory(self):
		"""
		Share memory for DMC v2.
		"""
		for position in self.positions:
			self.models[position].share_memory()

	def eval(self):
		"""
		Evaluate the model for DMC v2.
		"""
		for position in self.positions:
			self.models[position].eval()