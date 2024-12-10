# -*- coding: utf-8 -*-

import torch
import numpy as np


from predict.ddz.xxc.ddz_xxc_obs import get_obs
from predict.ddz.xxc.ddz_xxc_an_obs import get_obs_by_4_an


class DeepAgent:
	"""
	DeepAgent is a class for implementing a deep learning agent for the game of Dou di zhu.
	"""
	
	def __init__(self, position, model_path, is_4_an=False):
		"""
		Initialize the DeepAgent.
		
		:param position: int, the position of the agent in the game (0, 1, 2)
		:param model_path: str, the path to the pre-trained model
		:param is_4_an: bool, whether the game is 4 an or not
		"""
		self.__is_4_an = is_4_an
		self.position = position
		self.model_path = model_path
		self.model = self.init_model()

	def init_model(self):
		"""
		Initialize the deep learning model.
		"""
		if self.__is_4_an:
			from reinforce.model.ddz_xxc_an_models import an_model_dict as model_dict
		else:
			from reinforce.model.ddz_xxc_models import model_dict
		model = model_dict[self.position]()
		model_state_dict = model.state_dict()
		if torch.cuda.is_available():
			pretrained = torch.load(self.model_path, map_location='cuda:0')
		else:
			pretrained = torch.load(self.model_path, map_location='cpu')
		pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
		model_state_dict.update(pretrained)
		model.load_state_dict(model_state_dict)
		if torch.cuda.is_available():
			model.cuda()
		model.eval()
		return model

	def predict(self, info_set, must_play=False, index_list=None, uid=None):
		"""
		Predict the best action to take given the current information set.
		"""
		if len(info_set.legal_actions) == 1:
			return info_set.legal_actions[0], 0
		if not self.__is_4_an:
			obs = get_obs(info_set)
		else:
			obs = get_obs_by_4_an(info_set)
		z_batch = torch.from_numpy(obs['z_batch']).float()
		x_batch = torch.from_numpy(obs['x_batch']).float()
		if torch.cuda.is_available():
			z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
		y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
		y_pred = y_pred.detach().cpu().numpy()
		if must_play:
			best_action_index = self.fetch_must_play_index(y_pred, index_list)  # todo: 方块三先出
		else:
			best_action_index = np.argmax(y_pred, axis=0)[0]
		best_action = info_set.legal_actions[best_action_index]
		best_action_confidence = y_pred[best_action_index]
		return best_action, best_action_confidence

	@staticmethod
	def fetch_must_play_index(y_pred, index_list):
		"""
		params: y_pred: 预测值
		获取必须带的牌的组合最大的预测分 index
		"""
		must_play_list = []
		for i in index_list:
			must_play_list.append(y_pred[i])
		new_arr = np.array(must_play_list)
		best_action_index = np.argmax(new_arr, axis=0)[0]
		best_action_index = index_list[best_action_index]
		return best_action_index