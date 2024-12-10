# -*- coding: utf-8 -*-

import copy

from typing import Dict, Optional, Type
from reinforce.env.registry import EnvRegistry
from public.const import ENV_ENTRY_POINTS, DEFAULT_CONFIG


class EnvFactory:
	"""
	EnvFactory is a class that creates environments for the DeepMind Control Suite.
	"""

	def __init__(self, env_id: str) -> None:
		"""
		Initialize the EnvFactory with the environment ID.
		"""
		self.registry = EnvRegistry()
		self.env_infos = ENV_ENTRY_POINTS.get(env_id)
		self.env_id = self.env_infos.get("env_id")
		self.entry_point = self.env_infos.get("entry_point")

	def make(self, config: Dict[str, Optional[int | bool]] = None) -> Type[object]:
		"""
		Make the environment.
		"""
		if not config:
			config = copy.deepcopy(DEFAULT_CONFIG)

		# register the environment
		self.registry.register(self.env_id, self.entry_point)

		return self.registry.make(self.env_id, config)


def registration(env: str) -> Type[object]:
	"""
	Registration decorator for environments.
	"""
	return EnvFactory(env).make()