# -*- coding: utf-8 -*-

from rlcards.const.mahjong import const


# TODO: 鸡分计算(FK)
def other_ming_xi_data(lose_to=-1, score=0, card=0):
	"""
	其他明细数据
	"""
	data = {
		"lose_to": lose_to,
		"card": card,
		"score": score,
	}

	return data

def self_ming_xi_data(win_from=None, score=0, card=0) -> dict:
	"""
	当前明细数据
	"""
	data = {
		"win_from": [] if not win_from else win_from,
		"card": card,
		"score": score,
	}

	return data

def update_result_score(result: {}, seat_id, is_self, data=None):
	"""
	记录玩家结算明细，统一采用 加分形式
	res: account
	seat_id: 玩家座位号
	is_self: 0/1 0表示输,1表示赢(炸胡反过来)
	data: 明细分数
	"""
	score = data.get("score", 0)
	if score == 0:
		return
	if not result.get(seat_id):
		result[seat_id] = {
			"total_score": 0,
			"ming_xi": {"self": [], "other": []}
		}
	result[seat_id]["total_score"] += score
	if data:
		key = "self" if is_self else "other"
		result[seat_id]["ming_xi"].setdefault(key, []).append(data)

def compare_pai_type_is_gt_qing_yi_se(hu_type, is_zi_mo=False):
	"""
	牌型低于清一色算清一色，牌型高于清一色算实际牌型
	此接口只判断玩家牌型是否小于清一色
	"""
	pai_type_score = const.HU_PAI_SCORES.get(hu_type, 0)
	qing_yi_se_score = const.HU_PAI_SCORES.get(const.HuPaiType.QING_YI_SE)
	flag = True
	if pai_type_score < qing_yi_se_score:
		flag = False
		pai_type_score = qing_yi_se_score
	if is_zi_mo:
		pai_type_score *= 2
	return flag, pai_type_score

def card_count_fk(cards: list) -> dict:
	"""
	计算房卡场卡牌数量
	"""
	card_count = {}
	for c in cards:
		card_count[c] = card_count.get(c, 0) + 1
	return card_count