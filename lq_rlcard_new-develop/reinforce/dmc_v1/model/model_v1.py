# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch import nn
from torch.nn import LayerNorm

# Pytorch 2.x version

class Block(nn.Module):
	"""
	神经网络块
	"""

	def __init__(self, dim, layer_scale_init_value=1e-6):
		super(Block, self).__init__()
		# depth wise conv
		self.dw_conv1 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
		self.norm = LayerNorm(dim, eps=1e-6)

		# point-wise /1x1 conv, implemented with linear layers
		self.pw_conv1 = nn.Linear(dim, dim * 4)
		self.gelu = nn.GELU()
		self.pw_conv2 = nn.Linear(4 * dim, dim)
		self.gamma = nn.Parameter(
			layer_scale_init_value * torch.ones(dim),
			requires_grad=True
		) if layer_scale_init_value > 0 else None

	def forward(self, x):
		"""
		前向传播
		"""
		x = self.dw_conv1(x)
		x = x.permute(0, 2, 1)  # Adjusting for 1D: (N, C, L) -> (N, L, C)
		x = self.norm(x)
		x = self.pw_conv1(x)
		x = self.act(x)
		x = self.pw_conv2(x)
		if self.gamma is not None:
			x = self.gamma * x
		x = x.permute(0, 2, 1)  # Adjusting back: (N, L, C) -> (N, C, L)
		return x

class Bottleneck(nn.Module):
	"""
	Bottleneck块
	"""

	# 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(1,), bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,), stride=(stride,), padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)
		self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=(1,), bias=False)
		self.bn3 = nn.BatchNorm1d(self.expansion * planes)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion * planes, kernel_size=(1,), stride=(stride,), bias=False),
				nn.BatchNorm1d(self.expansion * planes)
			)

	def forward(self, x):
		"""
		前向传播
		"""
		out = torch.relu(self.bn1(self.conv1(x)))
		out = torch.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = torch.relu(out)
		return out


class BasicBlock(nn.Module):
	"""
	用于ResNet18和34的残差块，用的是2个3x3的卷积
	"""
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,), stride=(stride,), padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,), stride=(1,), padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)
		self.shortcut = nn.Sequential()
		# 经过处理后的x要与x的维度相同(尺寸和深度)，如果不相同，需要添加卷积+BN来变换为同一维度
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion * planes, kernel_size=(1,), stride=(stride,), bias=False),
				nn.BatchNorm1d(self.expansion * planes)
			)

	def forward(self, x):
		"""
		前向传播
		"""
		out = torch.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = torch.relu(out)
		return out

class ChannelAttention(nn.Module):
	"""
	通道注意力模块
	"""
	def __init__(self, channel, reduction=4):
		super().__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		"""
		前向传播
		"""
		b, c, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1)
		return x * y.expand_as(x)

class BasicBlockM(nn.Module):
	"""
	用于ResNet50、101和152的残差块，用的是3x3和1x1的卷积
	"""
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlockM, self).__init__()
		self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.mi_sh = nn.Mish(inplace=True)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)
		self.se = ChannelAttention(planes)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm1d(self.expansion * planes)
			)

	def forward(self, x):
		"""
		前向传播
		"""
		out = self.mi_sh(self.bn1(self.conv1(x)))
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.se(out)
		out += self.shortcut(x)
		out = self.mi_sh(out)
		return out

class GeneralModelResnet(nn.Module):
	"""
	ResNet模型
	"""
	def __init__(self):
		super().__init__()
		self.in_planes = 72
		self.layer1 = self._make_layer(BasicBlockM, 72, 3, stride=2)  # 1*27*72
		self.layer2 = self._make_layer(BasicBlockM, 144, 3, stride=2)  # 1*14*146
		self.layer3 = self._make_layer(BasicBlockM, 288, 3, stride=2)  # 1*7*292
		self.linear1 = nn.Linear(288 * BasicBlockM.expansion * 7 + 18 * 4, 2048)
		self.linear2 = nn.Linear(2048, 512)
		self.linear3 = nn.Linear(512, 128)
		self.linear4 = nn.Linear(128, 3)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, z, x, return_value=False, flags=None, debug=False):
		"""
		前向传播
		"""
		out = self.layer1(z)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.flatten(1, 2)
		out = torch.cat([x, x, x, x, out], dim=-1)
		out = torch.relu(self.linear1(out))
		out = torch.relu(self.linear2(out))
		out = torch.relu(self.linear3(out))
		out = self.linear4(out)
		if return_value:
			return dict(values=out)
		else:
			if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
				action = torch.randint(out.shape[0], (1,))[0]
			else:
				action = torch.argmax(out, dim=0)[0]
			return dict(action=action, max_value=torch.max(out), values=out)

class Model:
	"""
	模型
	"""

	def __init__(self, device=0):
		if not device == "cpu":
			device = 'cuda:' + str(device)
		position = ["landlord1", "landlord2", "landlord3", "landlord4"]
		self.models = {ps: GeneralModelResnet().to(torch.device(device)) for ps in position}

	def step(self, position, z, x, training=False, flags=None, debug=None):
		"""
		迭代更新
		"""
		model = self.models[position]
		return model.forward(z, x, training, flags, debug)

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