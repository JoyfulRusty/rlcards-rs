# -*- coding: utf-8 -*-

# todo: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
	"""
	M i s h activation function.
	"""

	def __init__(self):
		"""
		Initialize Mish object.
		"""
		super().__init__()

	@classmethod
	def mish(cls, input_val):
		"""
		mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
		"""
		if torch.__version__ >= "1.9":
			return F.mish(input_val)
		else:
			return input_val * torch.tanh(F.softplus(input_val))