# -*- coding: utf-8 -*-

from rlcards.predict.sytx.state import sy_state
from rlcards.predict.sytx.models import choice_model

def predict_action(data: dict):
	"""
	选择对应模型预测动作
	"""
	# 构建状态数据
	action_state = sy_state.extract_state(data)
	action = choice_model[data["self"]].step(action_state)
	return action