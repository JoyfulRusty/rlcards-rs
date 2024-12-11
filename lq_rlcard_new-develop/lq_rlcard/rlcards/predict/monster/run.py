# -*— coding: utf-8 -*-

from rlcards.predict.monster.state import predict_state
from rlcards.predict.monster.models import choice_model
from rlcards.games.monster.utils import new_card_encoding_dict


# 传入state = {}， 进行解析

def predict_action(data):
    """
    预测下一个动作
    """
    # 获取当前预测玩家的ID
    player_id = predict_state.make_state(data)
    print("读取解析后的数据: ", predict_state.new_state)

    # 获取打妖怪游戏中玩家的状态数据
    new_state = predict_state.encode_state

    # 选择模型
    model = choice_model.get(player_id)

    # 传入状态数据进行预测
    next_action_id = model.step(new_state, False)
    next_action = decode_action(next_action_id, data)

    return next_action

def decode_action(action_id, state):
    """
    解码动作
    """
    decode_action_id = {new_card_encoding_dict[key]: key for key in new_card_encoding_dict.keys()}
    action = decode_action_id[action_id]
    if action_id < 30:
        legal_cards = state['actions']
        for card in legal_cards:
            if card == action:
                action = card
                break

    return action