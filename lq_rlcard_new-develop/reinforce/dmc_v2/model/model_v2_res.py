# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch import nn

from reinforce.dmc_v2.mish import Mish

class BasicBlock(nn.Module):
	"""
	用于ResNet18和34的残差块，用的是2个3x3的卷积
	"""

	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		"""
		Initialize basic block
		:param in_planes: 输入通道数
		:param planes: 输出通道数
		:param stride: 步长
		"""
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,), stride=(stride,), padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,), stride=(1,), padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)
		self.shortcut = nn.Sequential()
		# 经过处理后的x要与x的维度相同(尺寸和深度)
		# 如果不相同，需要添加卷积+BN来变换为同一维度
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion * planes, kernel_size=(1,), stride=(stride,), bias=False),
				nn.BatchNorm1d(self.expansion * planes)
			)

	def forward(self, x):
		"""
		Forward the block
		"""
		val_x = self.conv1(x)
		val_x = self.bn1(val_x)
		val_x = Mish.mish(val_x)
		val_x = self.conv2(val_x)
		val_x = self.bn2(val_x)
		val_x += self.shortcut(x)
		val_x = Mish.mish(val_x)
		return val_x


class ResnetModel(nn.Module):
	"""
	残差模型
	"""

	def __init__(self, mlp_layers=None):
		"""
		Initialize the model
		:param mlp_layers: MLP层数
		"""
		super(ResnetModel, self).__init__()
		self.in_planes = 12
		if not mlp_layers:
			mlp_layers = [512, 512, 512, 512, 512]
		self.conv1 = nn.Conv1d(self.in_planes, 12, kernel_size=(3,), stride=(2,), padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(12)
		self.layer1 = self._make_layer(BasicBlock, 12, 2, stride=2)
		self.layer2 = self._make_layer(BasicBlock, 24, 2, stride=2)
		self.layer3 = self._make_layer(BasicBlock, 48, 2, stride=2)
		self.dense0 = nn.Linear(980, mlp_layers[0])
		self.dense1 = nn.Linear(mlp_layers[0], mlp_layers[1])
		self.dense2 = nn.Linear(mlp_layers[1], mlp_layers[2])
		self.dense3 = nn.Linear(mlp_layers[2], mlp_layers[3])
		self.dense4 = nn.Linear(mlp_layers[3], mlp_layers[4])
		self.dense5 = nn.Linear(mlp_layers[4], 1)

	def _make_layer(self, block, planes, num_blocks, stride):
		"""
		Make Net Layers
		"""
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, z, x):
		"""
		Forward the model.
		"""
		val_z = self.conv1(z)
		val_z = self.bn1(val_z)
		val_z = Mish.mish(val_z)
		val_z = self.layer1(val_z)
		val_z = self.layer2(val_z)
		val_z = self.layer3(val_z)
		val_z = val_z.flatten(1, 2)
		val_x = torch.cat([x, x, val_z], dim=-1)
		val_x = self.dense0(val_x)
		val_x = Mish.mish(val_x)
		val_x = self.dense1(val_x)
		val_x = Mish.mish(val_x)
		val_x = self.dense2(val_x)
		val_x = Mish.mish(val_x)
		val_x = self.dense3(val_x)
		val_x = Mish.mish(val_x)
		val_x = self.dense4(val_x)
		val_x = Mish.mish(val_x)
		val_x = self.dense5(val_x)
		return val_x


class Model:
	"""
	Model for DMC v2.
	"""

	def __init__(self, positions, mlp_layers=None, device="0"):
		"""
		初始化参数
		"""
		self.positions = positions
		self.mlp_layers = mlp_layers
		self.device = 'cuda:' + str(device) if device != "cpu" else "cpu"
		self.net = ResnetModel(mlp_layers=self.mlp_layers).to(torch.device(self.device))
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
		return legal_actions[action_idx]

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