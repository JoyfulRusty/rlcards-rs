# -*- coding: utf-8 -*-

import torch

import numpy as np

class DeepAgent:
	"""
	DeepAgent is a class that represents a deep learning agent.
	"""

	def __init__(self, position, model_path):
		"""
		Creates a new DeepAgent object.

		:param position: The position of the agent.
		:param model_path: The path to the model file.
		"""

		self.position = position
		self.model_path = model_path
		self.model = self.load_model()

	def load_model(self):
		"""
		Loads the model from the given path.
		"""
		model = None
		if torch.cuda.is_available():
			model = torch.load(self.model_path, map_location='cuda:0')
			model.cuda()
		else:
			model = torch.load(self.model_path, map_location='cpu')
		model.eval()
		return model

	def predict(self, z, x):
		"""
		Predicts the next state of the agent.
		"""
		y_pred = self.model.forward(z=z, x=x)
		y_pred = y_pred.cpu().detach().numpy()
		action_idx = np.argmax(y_pred, axis=0)[0]
		return action_idx