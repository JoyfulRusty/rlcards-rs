# -*- coding: utf-8 -*-

import collections

from predict.ddz.xxc.bid_agent import predict_bid_agent
from predict.ddz.xxc.farmer_agent import predict_farmer_agent

from predict.ddz.xxc.info_set import info_set
from predict.ddz.models import ADP_MODELS, BXP_MODELS, DEALER_MODEL_MAPS, WP_MODELS
from predict.ddz.xxc.const import *
from predict.ddz.xxc.rule import Rule


class Model:
	"""
	模型状态数据
	"""

	def __init__(self):
		"""
		初始化模型状态数据
		"""
		self.allow_act = {}
		self.curr_pid = None
		self.dealer_id = None
		self.curr_position = None
		self.cards_map = EXTRA_CARD_MAP.copy()

	def cal_play_card(self, uid, req_data: dict):
		"""
		计算出牌
		"""
		actions = []
		self.curr_pid = req_data.get("position")
		self.dealer_id = req_data.get("dealer_id")
		self.allow_act = req_data.get("allow_act", {})
		is_bxp = req_data.get("is_bxp")
		if is_bxp:
			positions = DEALER_MODEL_MAPS.get(self.dealer_id)
			self.curr_position = positions.get(self.curr_pid)
			model = BXP_MODELS.get(self.curr_position)
		else:
			positions = DEALER_MODEL_MAPS.get(self.dealer_id)
			self.curr_position = positions.get(self.curr_pid)
			if self.curr_pid > 2:
				model = WP_MODELS.get("landlord")
			else:
				model = ADP_MODELS.get(self.curr_position)
		play_cards = []
		self.get_info_set(req_data, is_bxp)
		curr_hand_cards = req_data.get("player_hand_cards") or []
		curr_hand_cards.sort()
		loop_hand_cards = curr_hand_cards[:]
		actions, win_rate = model.predict(info_set)
		actions_len = len(actions)
		for card in loop_hand_cards:
			c = card % 100
			if c in actions:
				actions.remove(c)
				play_cards.append(card)
			if len(play_cards) == actions_len or not actions:
				break
		# 判断鬼炸是否与其他炸弹💣组合
		if self.check_ghost_cards(play_cards):
			play_cards = self.judge_ghost_cards(self.curr_position, req_data, play_cards)
		print(f"tid: {req_data.get('tid')}, uid: {uid}, 出牌: {play_cards}")
		return play_cards

	@staticmethod
	def judge_ghost_cards(curr_pos, req_data, play_cards):
		"""
		todo: 必定满足鬼炸+1炸[四带二条件才会走此接口]
		判断鬼诈与其他牌型最佳组合
		"""
		# 判断鬼炸是否与其他牌型组合
		ghost_cards = []
		for card in play_cards:
			if card in GHOST_CARDS:
				ghost_cards.append(card)
		# todo: 必须为首出
		# 存在其他炸弹与鬼炸组合时，则重新计算最佳出牌
		other_p_cards_nums = req_data.get('cards_left_dict', {})
		for pos, nums in other_p_cards_nums.items():
			if pos == curr_pos:
				continue
			# 其他玩家可能存在关门炸弹💣，先出鬼炸
			if nums == 4:
				return ghost_cards
			other_p_cards_nums[curr_pos] = 0
			count_nums = sum(other_p_cards_nums.values())
			# 当其他玩家出较少牌时，先出鬼炸
			if count_nums > 33:
				return ghost_cards
			# 先出较小炸弹💣，鬼炸最后出
			for card in ghost_cards:
				play_cards.remove(card)
			return play_cards

	@staticmethod
	def check_ghost_cards(play_cards):
		"""
		检测当前出牌是否为鬼炸加其他类型炸弹
		"""
		# 当前可选出牌数量不为6时
		if len(play_cards) != 6:
			return False
		ghost_cards = []
		# 拷贝当前可选合法出牌
		tmp_play_cards = play_cards[:]
		for card in play_cards:
			if card in GHOST_CARDS:
				ghost_cards.append(card)
				tmp_play_cards.remove(card)
		# 判断移除大小鬼之后，炸弹条件是否满足
		tmp_play_cards = [card % 100 for card in tmp_play_cards]
		# 判断计算大小鬼+1炸弹💣条件是否满足
		if len(ghost_cards) == 2 and len(set(tmp_play_cards)) > 0:
			return True
		return False

	@staticmethod
	def get_comb_by_3():
		"""
		获取3带1组合索引
		"""
		return [idx for idx, actions in enumerate(info_set.legal_actions) if actions[0] == 3]

	def get_info_set(self, req_data: dict, is_bxp=False):
		"""
		收集玩家可观测数据
		"""
		info_set.bomb_num = req_data.get("bomb_num", 0)
		card_play_action_seq = req_data.get("action_seq", [])
		info_set.last_move = self.get_last_move(card_play_action_seq)
		info_set.last_two_moves = self.get_last_two_move(card_play_action_seq)
		info_set.last_move_dict = req_data.get("last_move_dict", {})
		last_pid = self.curr_pid - 1 if self.curr_pid != 0 else 2
		info_set.last_pid = ALL_POSITION[last_pid]
		player_hand_cards = [card % 100 for card in req_data.get("player_hand_cards")] or []
		info_set.legal_actions = Rule.get_legal_card_play_actions(
			cards=player_hand_cards,
			action_sequence=card_play_action_seq,
			allow_actions=self.allow_act
		)
		info_set.num_cards_left_dict = req_data.get("cards_left_dict")
		info_set.played_cards = req_data.get("played_cards", {})
		info_set.player_hand_cards = player_hand_cards
		self.get_other_hand()  # 所有的牌 - 自己手牌 - 已经出了的牌
		info_set.player_position = self.curr_position
		info_set.card_play_action_seq = card_play_action_seq

	def get_left_cards(self, all_hand_cards):
		"""
		获取剩余手牌
		"""
		num_cards_left = {}
		for k in all_hand_cards:
			cards_len = len(all_hand_cards.get(k, []))
			if k != self.curr_position:
				num_cards_left[k] = cards_len if cards_len <= CARD2POINT else CARD2POINT
				continue
			num_cards_left[k] = cards_len
		return num_cards_left

	@staticmethod
	def get_last_move(card_play_action_seq):
		"""
		获取上家出牌
		"""
		last_move = []
		if len(card_play_action_seq) != 0:
			if len(card_play_action_seq[-1]) == 0:
				last_move = card_play_action_seq[-2]
			else:
				last_move = card_play_action_seq[-1]
		return last_move

	@staticmethod
	def get_last_move_an_4(card_play_action_seq):
		"""
		获取4人暗地主上一个动作，不包含pass
		"""
		last_move = []
		action_sequence = card_play_action_seq
		if len(action_sequence) != 0:
			if len(action_sequence[-1]) == 0:
				if len(action_sequence[-2]) != 0:
					last_move = action_sequence[-2]
				else:
					last_move = action_sequence[-3]
			else:
				last_move = action_sequence[-1]
		return last_move

	@staticmethod
	def get_last_two_move(card_play_action_seq):
		""" 最新的两次动作(出牌) """
		last_two_moves = [[], []]
		for card in card_play_action_seq[-2:]:
			last_two_moves.insert(0, card)
			last_two_moves = last_two_moves[:2]
		return last_two_moves

	@staticmethod
	def value(cards: list):
		values = []
		for c in cards:
			values.append(c % 100)
		return values

	@staticmethod
	def get_other_hand():
		"""
		获取出当前玩家手牌以及已经出过的牌的剩余手牌
		"""
		info_set.other_hand_cards = []
		played_cards_tmp = []
		for i in list(info_set.played_cards.values()):
			played_cards_tmp.extend(i)
		# 出过的牌和玩家手上的牌
		played_and_hand_cards = played_cards_tmp + info_set.player_hand_cards
		# 整副牌减去出过的牌和玩家手上的牌，就是其他人的手牌
		for i in set(AllEnvCard):
			info_set.other_hand_cards.extend(
				[i] * (AllEnvCard.count(i) - played_and_hand_cards.count(i)))

	@staticmethod
	def exchange_in_three(cards):
		"""
		换三张
		采取评分策略(只考虑手牌，不考虑队友)
		"""
		cards_val = cards[:]
		cards_dic = collections.defaultdict(int)
		for v in cards_val or []:
			cards_dic[v] += 1
		record_score = {}.fromkeys(cards_dic.keys(), 0)  # 计分器
		for k, v in record_score.items():
			record_score[k] += BASE_SCORE.get(k, 0)
		if S_KING in cards_dic and B_KING in cards_dic:  # 大小王
			record_score[S_KING] += GradeScore.ZHA_DAN * 4
			record_score[B_KING] += GradeScore.ZHA_DAN * 4
		elif S_KING in cards_dic:
			record_score[S_KING] += GradeScore.XIAO_WANG
		elif B_KING in cards_dic:
			record_score[B_KING] += GradeScore.DA_WANG

		if CARD2POINT in cards_dic:
			record_score[CARD2POINT] += GradeScore.VAL_2
		pairs = []
		three = []
		# 炸弹一般不换
		for k, v in cards_dic.items():
			if v == 4:
				record_score[k] += GradeScore.ZHA_DAN * v
			elif v >= 2:
				v == 3 and three.append(k)
				pairs.append(k)
		# 单顺子
		dan_shun = Rule.get_shun_zi(list(cards_dic.keys()), min_num=5)
		for shun in dan_shun or []:
			inter_3 = len(list(set(shun).intersection(three)))
			inter_2 = list(set(shun).intersection(pairs))  # 计算重合部分
			inter_2_len = len(inter_2)
			inter_2_3 = len(list(set(three).intersection(pairs)))
			inter_2_len -= inter_2_3
			lose_fj = GradeScore.SHUN_ZI - inter_3 * GradeScore.DAN_FEI_JI - inter_2_len * GradeScore.PAIRS
			for s in shun:
				record_score[s] += lose_fj
		# 连对
		combo_shun = pairs and Rule.get_shun_zi(pairs, min_num=3)
		for shun in combo_shun or []:
			for s in shun:
				record_score[s] += GradeScore.LIAN_DUI * 2
		# 飞机
		san_shun = three and Rule.get_shun_zi(three, min_num=2)
		for shun in san_shun or []:
			for s in shun:
				three.remove(s)
				record_score[s] += GradeScore.FEI_JI * 3

		for t in three:
			record_score[t] += GradeScore.DAN_FEI_JI * 3

		exchange_val = []
		for _ in range(3):
			k = min(record_score, key=record_score.get)
			if cards_dic[k] > 1:
				cards_dic[k] -= 1
			else:
				record_score.pop(k)
			exchange_val.append(k)

		exchange_cards = []
		for c in cards:
			c_val = c % 100
			if c_val in exchange_val:
				exchange_val.remove(c_val)
				exchange_cards.append(c)
		return exchange_cards

	@staticmethod
	def exchange_in_five(cards, val_threshold=12, need_len=5):
		"""
		4人暗地主换5张
		"""
		cards.sort()
		cards_dic = collections.defaultdict(int)
		for v in cards or []:
			cards_dic[v] += 1
		exchange_val = []
		cards_val_copy = cards[:]
		for c_val in cards:
			count = cards_dic.get(c_val, 0)
			if count == 1 and c_val <= val_threshold:
				exchange_val.append(c_val)
				cards_val_copy.remove(c_val)

		dan_len = len(exchange_val)
		if dan_len > 5:
			exchange_val = exchange_val[:need_len]
		elif dan_len < 5:
			cards_val_copy.sort()
			exchange_val += cards_val_copy[:need_len - dan_len]

		exchange_cards = []
		for c in cards:
			c_val = c % 100
			if c_val in exchange_val:
				exchange_val.remove(c_val)
				exchange_cards.append(c)
		return exchange_cards

	def cal_play_bid(self, hand_cards, can_bids, is_bxp=False):
		""" 叫分 """
		if not can_bids:
			can_bids = [1]
		cards = [card % 100 for card in hand_cards]
		win_rate = predict_bid_agent.predict_score(cards)
		win_rate = round(win_rate, 3)
		farmer_score = predict_farmer_agent.predict(cards, "farmer")  # 游戏前预估得分
		compare_winrate = win_rate
		if compare_winrate > 0:
			compare_winrate *= 2.5
		landlord_requirement = True
		if USE_RULE_LANDLORD_REQUIREMENTS:
			landlord_requirement = self.rule_landlord_requirement(cards)
		if is_bxp:
			bid_thresholds = BID_THRESHOLDS_AN
			bid_thresholds_by_winrate = BID_THRESHOLDS_BY_AN_WINRATE
			print(f"不洗牌叫分信息: {cards}-{win_rate}-{farmer_score}")
		else:
			bid_thresholds = BID_THRESHOLDS
			bid_thresholds_by_winrate = BID_THRESHOLDS_BY_WINRATE
		index = 0 if len(can_bids) == 3 else 1
		if win_rate > bid_thresholds[index] and compare_winrate > farmer_score and landlord_requirement:
			return self.get_bid_score(win_rate, bid_thresholds_by_winrate)  # 叫分
		else:
			return 0

	@staticmethod
	def get_bid_score(win_rate, bid_thresholds_by_winrate: dict):
		""" 根据手牌胜率叫分 """
		score = 0
		for s, t in bid_thresholds_by_winrate.items():
			if win_rate > t:
				score = s
		return score

	@staticmethod
	def cal_play_chan(hand_cards, chan_threshold=None):
		if not chan_threshold:
			chan_threshold = CHAN_THRESHOLDS[1][0]
		cards = [card % 100 for card in hand_cards]
		farmer_score = predict_farmer_agent.predict(cards, "farmer")  # 游戏前预估得分
		win_rate = round(farmer_score, 3)
		if win_rate > chan_threshold:
			return 1
		return 0

	@staticmethod
	def rule_landlord_requirement(cards):
		"""
		基于规则(经验)判断是否要叫地主
		cards: 牌值
		"""
		cards_count = {}
		for c in cards:
			cards_count[c] = cards_count.get(c, 0) + 1
		if (cards_count[B_KING] == 1 and cards_count[CARD2POINT] >= 1 and cards_count[14] >= 1) \
			or (cards_count[B_KING] == 1 and cards_count[CARD2POINT] >= 2) \
			or (cards_count[B_KING] == 1 and len([key for key in cards_count if cards_count[key] == 4]) >= 1) \
			or (cards_count[B_KING] == 1 and cards_count[S_KING] == 1) \
			or (len([key for key in cards_count if cards_count[key] == 4]) >= 2) \
			or (cards_count[S_KING] == 1 and (
			(cards_count[CARD2POINT] >= 2) or (cards_count[CARD2POINT] >= 2 and cards_count[14] >= 2) or (
			cards_count[CARD2POINT] >= 2 and len([key for key in cards_count if cards_count[key] == 4]) >= 1))) \
			or (cards_count[CARD2POINT] >= 2 and len([key for key in cards_count if cards_count[key] == 4]) >= 1):
			return True
		else:
			return False