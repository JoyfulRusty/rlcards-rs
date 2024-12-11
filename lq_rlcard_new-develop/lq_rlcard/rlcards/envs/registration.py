# -*- coding: utf-8 -*-

import importlib

# 默认配置
DEFAULT_CONFIG = {
    'allow_step_back': False,  # 是否允许step返回，默认为False
    'seed': None  # seed默认为None
}

class EnvSpec:
    """
    特定环境实例的规范
    """
    def __init__(self, env_id, entry_point=None):
        """
        初始化环境实例
        """
        self.env_id = env_id
        mod_name, class_name = entry_point.split(':')  # 按':'进行分割
        self._entry_point = getattr(importlib.import_module(mod_name), class_name)

    def make(self, config=None):
        """
        构建环境
        """
        if config is None:
            config = DEFAULT_CONFIG
        env = self._entry_point(config)
        return env

class EnvRegistry:
    """
    注册游戏环境
    """
    def __init__(self):
        """
        初始化参数
        """
        self.env_specs = {}

    def register(self, env_id, entry_point):
        """
        注册一个环境
        """
        if env_id in self.env_specs:
            raise ValueError("Cannot re-register env_id: {}".format(env_id))
        self.env_specs[env_id] = EnvSpec(env_id, entry_point)

    def make(self, env_id, config=None):
        """
        创建环境实列
        """
        if config is None:
            config = DEFAULT_CONFIG
        if env_id not in self.env_specs:
            raise ValueError("Cannot find env_id: {}".format(env_id))
        return self.env_specs[env_id].make(config)

# 设置全局注册表，环境注册
registry = EnvRegistry()

def register(env_id, entry_point):
    """
    注册环境ID
    """
    return registry.register(env_id, entry_point)

def make(env_id, config=None):  # env_id: 'mahjong_bak', configs: {'seed': 2022}
    """
    通过环境ID，构建环境实例
    """
    if config is None:
        config = {}
    _config = DEFAULT_CONFIG.copy()  # _config: {'allow_step_back': False, 'seed': None}
    for key in config:  # key: 'seed'
        _config[key] = config[key]

    return registry.make(env_id, _config)