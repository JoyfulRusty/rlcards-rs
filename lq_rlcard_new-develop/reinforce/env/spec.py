# -*- coding: utf-8 -*-

import importlib

from typing import Dict, Optional, Type


class EnvSpec:
	"""
	Class for specifying environments.
	"""

	def __init__(self, env_id: str, entry_point: str) -> None:
		"""
		Initializes the EnvSpec object.
		"""
		mod_name: str
		class_name: str
		self.env_id: str = env_id
		mod_name, class_name = entry_point.split(":")
		# This is in case you have a custom entry point
		self._entry_point = getattr(importlib.import_module(mod_name), class_name)

	def make(self, config: Dict[str, Optional[int | bool]]) -> Type[object]:
		"""
		Makes the environment.
		"""
		return self._entry_point(config)