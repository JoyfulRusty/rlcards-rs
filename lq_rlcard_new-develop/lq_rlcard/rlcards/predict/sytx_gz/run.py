# -*- coding: utf-8 -*-

from rlcards.const.sytx_gz.const import ActionType
from rlcards.predict.sytx_gz.state import sy_state
from rlcards.predict.sytx_gz.models import model_0, model_1

def predict_farmer(data: dict):
	"""
	todo: 闲家动作预测模型
	"""
	# 输出当前预测动作ID
	new_state = sy_state.build_state(data)
	# print("构建的神经网络模型预测数据: ", new_state)
	state = sy_state.extract_state(new_state)
	# 传入状态数据至神经网络模型中
	action_id = model_1.step(state)
	action = sy_state.decode_action_id[action_id]
	action_command = calc_commands(new_state, action)

	return action, action_command

def predict_landlord(data: dict):
	"""
	todo: 庄家动作预测模型
	"""
	# 输出当前预测动作ID
	new_state = sy_state.build_state(data)
	# print("构建的神经网络模型预测数据: ", new_state)
	state = sy_state.extract_state(new_state)
	# 传入状态数据至神经网络模型中
	action_id = model_0.step(state)
	action = sy_state.decode_action_id[action_id]
	action_command = calc_commands(new_state, action)

	return action, action_command

def calc_commands(new_state, action):
	"""
	解析动作调用命令
	"""
	if action in [ActionType.SP, ActionType.QG, ActionType.MI]:
		return 37
	if new_state["last_action"] in [ActionType.SP, ActionType.MI, ActionType.ZOU]:
		return {
			ActionType.SHA: 26,
			ActionType.ZOU: 27,
			ActionType.KAI: 31,
			ActionType.REN: 32,
			ActionType.XIN: 39,
			ActionType.FAN: 40
		}[action]
	return {ActionType.XIN: 29, ActionType.FAN: 28}[action]