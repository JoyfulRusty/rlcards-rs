# -*— coding: utf-8 -*-

from rlcards.predict.pig.make_obs import info_set
from rlcards.predict.pig.agent import choice_model
from rlcards.predict.pig.utils import get_obs


def predict_action(data):
    """
    TODO: 预测模型输出下一个动作
    """
    # 构建更新state
    print("更新数据: ", data)
    info_set.update_state(data)
    # 获取编码后的obs
    obs = get_obs(info_set)

    # 根据位置选择模型
    model = choice_model.get(info_set.player_position)
    # 输出最佳动作
    best_action = model.act(obs)
    return best_action[0]