# -*- coding: utf-8 -*-

from typing import Dict, Type
from reinforce.env.spec import EnvSpec

class EnvRegistry:
	"""
	A registry for environments.
	"""

	def __init__(self) -> None:
		"""
		Initializes the registry.
		"""
		self.env_specs: Dict[str, EnvSpec] = {}

	def register(self, env_id: str, entry_point: str) -> None:
		"""
		Registers an environment.
		"""
		self.env_specs[env_id]: Dict[str, EnvSpec] = EnvSpec(env_id, entry_point)

	def make(self, env_id: str, config: Dict[str, int | bool]) -> Type[object]:
		"""
		Makes an environment.
		"""
		return self.env_specs[env_id].make(config)