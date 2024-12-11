# -*- coding: utf-8 -*-

import itertools
import numpy as np

from collections import Counter, OrderedDict

from rlcards.games.sytx_gz.rules import Rules
from rlcards.games.mahjong.utils import random_choice_num
from rlcards.games.sytx_gz.allocat import get_better_combine_by_3
from rlcards.const.sytx_gz.const import FarmerAction, LandLordAction
from rlcards.games.sytx_gz.utils import new_card_encoding_dict, new_action_encoding_dict


class SyState:
	"""
	水鱼状态编码
	"""
	def __init__(self):
		"""
		初始化参数
		"""
		self.action_id = new_action_encoding_dict
		self.decode_action_id = {self.action_id[key]: key for key in self.action_id.keys()}
		self.history_rounds = []

	def build_state(self, state):
		"""
		todo: 构建AI预测单回合对局状态数据
		"""
		new_state = {
			"self": state["seat_id"],
			"role": state["role"],
			"is_winner": state["is_winner"],
			"bright_cards": state["bright_cards"],
			"combine_cards": get_better_combine_by_3(state["bright_cards"]),
			"other_hand_cards": state["other_hand_cards"],
			"remain_cards": state["remain_cards"],
			"history_rounds": state["history_round"],
			"history_actions": self.decode_history_actions(state["history_actions"]),
			"compare_big_cards": state["compare_big_cards"],
			"actions": self.calc_legal_by_actions(state),
			"last_action": self.decode_update_actions(state["last_action"]),
			"round_actions": self.decode_update_actions(state["round_actions"])
		}

		return new_state

	def calc_legal_by_actions(self, state):
		"""
		todo: 水鱼AI动作选择
		"""
		# 更新合法动作和上一个动作
		actions = self.decode_update_actions(state["actions"])
		last_action = self.decode_update_actions(state["last_action"])

		# 分扑已可组合的大扑
		combine_cards = get_better_combine_by_3(state["bright_cards"])
		div_cards = [card % 100 for card in combine_cards[:2]]

		# 强攻或密特殊处理
		if not last_action:
			# 此处单独计算闲家的撒扑操作选择
			if len(actions) > 1:
				if len(set(div_cards)) == 1 and self.calc_one_card_by_value(combine_cards[-1]):
					print("闲家存在豹子，单张也符合强攻或密: ", div_cards)
					actions.remove(FarmerAction.SP)
					return actions
				# 否则只能撒扑，删除强攻或密操作
				actions.remove(FarmerAction.QG)
				actions.remove(FarmerAction.MI)
				return actions
			# 此处计算庄家那个说话撒扑
			return actions

		# 闲家密，庄家有豹则必开
		elif last_action == FarmerAction.MI:
			print("闲密庄必开: ", div_cards)
			actions.remove(LandLordAction.REN)
			return actions

		# 庄杀闲必开
		elif last_action == LandLordAction.SHA:
			print("庄杀闲家必反: ", div_cards)
			actions.remove(FarmerAction.XIN)
			return actions

		# todo: 暂时取消此部分逻辑判断，模型自动拟合动作
		# # 庄走按条件判断选择动作
		elif last_action == LandLordAction.ZOU:
			# 计算豹子和剩余点数是否满足必反条件
			if len(set(div_cards)) == 1:
				if self.calc_one_card_by_value(combine_cards[-1]):
					print("庄家走，闲家豹子且条件满足反操作: ", div_cards)
					actions.remove(FarmerAction.XIN)
					return actions
			# 计算组合卡牌点数是否满足必须反条件
			res = self.calc_div_cards_by_value(div_cards)
			if res:
				if self.calc_one_card_by_value(combine_cards[-1]):
					print("庄家走，闲家已组合的扑点大于7且条件满足操作: ", div_cards)
					actions.remove(FarmerAction.XIN)
					return actions
			print("庄家走，上述条件不满足，则根据概率来对动作进行选择")
			action = self.calc_landlord_zou_by_actions([card % 100 for card in combine_cards[:2]])
			actions.remove(action)
			return actions

		# todo: 闲家撒扑，判断庄家操作
		elif last_action == FarmerAction.SP:
			farmer_cards = get_better_combine_by_3(state["other_hand_cards"])
			compare_res = self.calc_landlord_by_sha(farmer_cards, combine_cards)
			# 大于，则可对杀操作选择有一定概率
			if compare_res:
				print("庄家已组合扑大于闲家，存在杀概率选择: ", compare_res)
				if self.calc_one_card_by_value(combine_cards[-1]):
					print("闲家撒扑，庄家其中一扑大于闲家，且判断条件满足必杀1~")
					actions.remove(LandLordAction.ZOU)
					return actions
				# 判断闲家组合扑是否为豹子或小于3点
				decode_cards = [card % 100 for card in farmer_cards[:2]]
				# 闲家组合扑不为豹子，且点数小于4，庄家组合扑为豹子，庄必杀
				if len(set(decode_cards)) != 1 and len(set(combine_cards[:2])) == 1:
					for card in decode_cards:
						if card > 9:
							decode_cards.remove(card)
					decode_res = sum(decode_cards)
					if decode_res > 9:
						decode_res = decode_res - 10
					if decode_res < 4:
						print("闲家撒扑，庄家其中一扑大于闲家，且判断条件满足必杀2~")
						actions.remove(LandLordAction.ZOU)
						return actions
				return actions
			# 小于，则直接赛选掉可选的杀操作
			print("庄家已组合扑小于闲家，则必走: ", compare_res)
			actions.remove(LandLordAction.SHA)
			return actions
		return actions

	def calc_landlord_zou_by_actions(self, div_cards: list):
		"""
		庄家走时，计算闲家操作
		"""
		scope = [0.65, 0.35]
		if len(set(div_cards)) == 1:
			# 设置概率
			scope = [0.4, 0.6]
			action_id = int(random_choice_num([FarmerAction.FAN, FarmerAction.XIN], scope))
			decode_id = self.action_id[action_id]
			print("庄家走，闲家已确定的组合扑为豹子，优先删除操作: ", self.decode_action_id[decode_id])
			return action_id
		is_gt_9 = False
		for card in div_cards:
			if card > 9:
				div_cards.remove(card)
				is_gt_9 = True
		res = sum(div_cards)
		if res > 9:
			res = res - 10
		if is_gt_9 and res > 7:
			action_id = int(random_choice_num([FarmerAction.FAN, FarmerAction.XIN], scope))
			decode_id = self.action_id[action_id]
			print("庄家走，闲家已确定的组合扑为花色+大于7点，优先删除操作: ", self.decode_action_id[decode_id])
			return action_id
		elif res > 8:
			# 设置概率
			scope = [0.5, 0.5]
			action_id = int(random_choice_num([FarmerAction.FAN, FarmerAction.XIN], scope))
			decode_id = self.action_id[action_id]
			print("庄家走，闲家已确定的组合扑大于8点，优先删除操作: ", self.decode_action_id[decode_id])
			return action_id
		# 上述条件不满足时，优先删除反操作，大概率选择信操作
		if res < 5:
			# 设置概率
			scope = [0.7, 0.3]
		action_id = int(random_choice_num([FarmerAction.FAN, FarmerAction.XIN], scope))
		decode_id = self.action_id[action_id]
		print("庄家走，闲家组合扑不好，优先删除操作: ", self.decode_action_id[decode_id])
		return action_id

	@staticmethod
	def calc_div_cards_by_value(div_cards):
		"""
		计算组合值
		"""
		# 是否为豹子
		scope = [0.5, 0.5]
		if len(list(set(div_cards))) == 1:
			res = bool(random_choice_num([False, True], scope))
			print("当前组合卡牌为豹子: ", res)
			return res

		# 计算花色+牌值
		is_hua = False
		is_seven = 0
		for card in div_cards:
			if card > 9:
				is_hua = True
			if 7 < card <= 9:
				is_seven = card
		# 是否满足花色+大点
		if is_hua and is_seven:
			if is_seven == 9:
				scope = [0.4, 0.6]
			res = bool(random_choice_num([False, True], scope))
			print("当前卡牌组合为花色+大于7点: ", res)
			return res

		# 非花色是大点牌数
		for i in div_cards:
			if i > 9:
				div_cards.remove(i)
		res = sum(div_cards)
		# 计算最后点数值大小
		if res > 9:
			res = res - 10
		return False if res < 8 else True

	@staticmethod
	def calc_one_card_by_value(res_card):
		"""
		计算剩余一张卡牌是值大小
		"""
		res_value = res_card % 100
		if res_value < 3:
			res = bool(random_choice_num([False, True], [0.6, 0.4]))
			print("当前未组合扑的单张卡牌值小于3时，选择操作: ", res)
			return res
		elif 3 < res_value < 6:
			res = bool(random_choice_num([False, True], [0.6, 0.4]))
			print("当前未组合扑的单张卡牌值在[3, 6]区间时，选择操作: ", res)
			return res
		elif res_value > 9:
			res = bool(random_choice_num([False, True], [0.55, 0.45]))
			print("当前未组合扑的单张卡牌为花色时，选择操作: ", res)
			return res
		res = bool(random_choice_num([False, True], [0.7, 0.3]))
		print("当前未组合扑的单张卡牌值大于6时，选择操作: ", res)
		return res

	@staticmethod
	def calc_landlord_by_sha(farmer_cards, landlord_cards):
		"""
		闲家撒扑，机器人庄家必杀条件
		"""
		if len(farmer_cards) < 3 or len(landlord_cards) < 3:
			return False
		# 比较闲庄已确定组合的大小扑值大小
		res = Rules.compare_one_combine(farmer_cards[:2], landlord_cards[:2])
		if res not in [3, 4]:
			return False
		return False if res == 3 else True  # 3大4小

	@staticmethod
	def compare_one(farmer_cards, landlord_cards):
		"""
		比较剩余的一张
		"""
		compare_res = True if farmer_cards[-1] < landlord_cards[-1] else False
		if not compare_res:
			return False
		return True if 4 < landlord_cards[-1] % 100 < 9 else False

	def decode_history_actions(self, history_actions):
		"""
		解码历史对局序列动作
		"""
		if not history_actions:
			return []
		cache_history_actions = []
		for history_action in history_actions:
			cache_history_actions.append(self.decode_update_actions(history_action))
		return cache_history_actions

	@staticmethod
	def decode_update_actions(actions):
		"""
		todo: 解码更新动作
		"""
		decoded_actions = []
		dict_actions = {
			1: FarmerAction.SP,  # 闲家信
			2: FarmerAction.QG,  # 闲家强攻
			3: FarmerAction.MI,  # 闲家密
			4: FarmerAction.XIN,  # 闲家信
			5: FarmerAction.FAN,  # 闲家反
			6: LandLordAction.REN,  # 庄家认
			7: LandLordAction.KAI,  # 庄家开
			8: LandLordAction.SHA,  # 庄家杀
			9: LandLordAction.ZOU  # 庄家走
		}
		if actions is None:
			return None

		# 解码记录的上一个动作
		if isinstance(actions, int):
			return dict_actions[actions]

		# 解码当前可选的合法动作
		for action in actions:
			decoded_actions.append(dict_actions[action])

		return decoded_actions

	def extract_state(self, state):
		"""
		抽取状态编码
		"""
		# 状态编码state
		obs = self.get_obs(state)
		extracted_state = {
			'obs': obs,
			'z_obs': obs.reshape(9, 427).astype(np.float32),
			'actions': self.get_legal_actions(state),
			'row_state': state,
			'row_legal_actions': self.get_row_legal_actions(state['actions']),
		}

		return extracted_state

	def get_obs(self, state):
		"""
		根据当前角色选择状态编码
		"""
		if state["role"] == "farmer":
			return self.get_farmer_obs(state)
		return self.get_landlord_obs(state)

	def get_legal_actions(self, state):
		"""
		计算合法动作
		"""
		legal_action_id = {}
		legal_actions = state["actions"]
		if legal_actions:
			for action in legal_actions:
				legal_action_id[self.action_id[action]] = None
		return OrderedDict(legal_action_id)

	@staticmethod
	def get_row_legal_actions(actions):
		"""
		合法动作
		"""
		if not actions:
			return []
		return [action for action in actions]

	def decode_actions(self, state, action_id):
		"""
		解码动作
		"""
		if isinstance(action_id, dict):
			return
		decode_action = self.decode_action_id[action_id]
		if action_id < 10:
			legal_actions = self.get_legal_actions(state)
			for action in legal_actions:
				if action == decode_action:
					decode_action = action
					break
		return decode_action

	def get_landlord_obs(self, state):
		"""
		庄家单局回合状态编码
		"""
		curr_hand_cards = self.encode_cards(state['bright_cards'])
		curr_big_and_small_cards = self.encode_big_or_small_bright_cards(state['bright_cards'])
		curr_combine_cards = self.encode_cards(state['combine_cards'])
		compare_big_cards = self.compare_farmer_and_landlord_big_cards(state["compare_big_cards"])
		other_hand_cards = self.encode_cards(state['other_hand_cards'])
		other_big_and_small_cards = self.encode_big_or_small_bright_cards(state['other_hand_cards'])
		remain_cards = self.encode_cards(state['remain_cards'])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		history_actions = self.encode_history_actions(state['history_actions'])
		action_roles = self.encode_landlord_or_farmer(state['role'])

		# 对一整局游戏所使用的状态信息编码，十局后则清空，重新记录
		history_rounds = self.encode_history_round(state["history_rounds"])

		obs = np.hstack((
			curr_hand_cards,
			curr_big_and_small_cards,
			curr_combine_cards,
			compare_big_cards,
			other_hand_cards,
			other_big_and_small_cards,
			remain_cards,
			last_action,
			legal_actions,
			round_actions,
			history_actions,
			action_roles,
			history_rounds
		))

		return obs

	def get_farmer_obs(self, state):
		"""
		闲家单局回合状态编码
		"""
		curr_hand_cards = self.encode_cards(state['bright_cards'])
		curr_big_and_small_cards = self.encode_big_or_small_bright_cards(state['bright_cards'])
		curr_combine_cards = self.encode_cards(state['combine_cards'])
		compare_big_cards = self.compare_farmer_and_landlord_big_cards(state["compare_big_cards"])
		other_hand_cards = self.encode_cards(state['other_hand_cards'])
		other_big_and_small_cards = self.encode_big_or_small_bright_cards(state['other_hand_cards'])
		remain_cards = self.encode_cards(state['remain_cards'])
		last_action = self.encode_last_action(state['last_action'])
		legal_actions = self.encode_legal_actions(state['actions'])
		round_actions = self.encode_round_actions(state['round_actions'])
		history_actions = self.encode_history_actions(state['history_actions'])
		action_roles = self.encode_landlord_or_farmer(state['role'])

		# 对一整局游戏所使用的状态信息编码，十局后则清空，重新记录
		history_rounds = self.encode_history_round(state["history_rounds"])

		obs = np.hstack((
			curr_hand_cards,
			curr_big_and_small_cards,
			curr_combine_cards,
			compare_big_cards,
			other_hand_cards,
			other_big_and_small_cards,
			remain_cards,
			last_action,
			legal_actions,
			round_actions,
			history_actions,
			action_roles,
			history_rounds
		))

		return obs

	@staticmethod
	def combine_bright_cards(bright_cards):
		"""
		明牌x3组合
		"""
		combine_cards = itertools.permutations(bright_cards, 3)
		new_combine_cards = [list(cards) for cards in combine_cards]
		return new_combine_cards

	def encode_big_or_small_bright_cards(self, bright_cards):
		"""
		编码亮牌3X3组合
		"""
		if not bright_cards:
			return np.zeros(288, dtype=np.float32)

		# 三种组合，编码成9张
		all_combine_matrix = np.zeros((6, 48), dtype=np.float32)

		# 使用[4 x 12]矩阵来进行编码
		combine_cards = self.combine_bright_cards(bright_cards[:3])
		for idx, cards in enumerate(combine_cards):
			matrix = np.zeros((4, 12), dtype=np.float32)
			cards_dict = Counter(cards)
			for card, num_times in cards_dict.items():
				matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1

			all_combine_matrix[idx] = matrix.flatten('A')

		return all_combine_matrix.flatten('A')

	@staticmethod
	def compare_farmer_and_landlord_big_cards(compare_big_cards):
		"""
		todo: 比较亮牌后，庄闲家确定大扑大小比较
		"""
		if not compare_big_cards or compare_big_cards == "draw":
			return np.zeros(18, dtype=np.float32)
		matrix = np.zeros((2, 9), dtype=np.float32)
		# 判断庄闲大小扑谁大，根据庄闲ID编码
		# 庄家大扑大
		if compare_big_cards == "landlord":
			matrix[0][:] = 1

		# 闲家大扑大
		elif compare_big_cards == "farmer":
			matrix[1][:] = 1

		return matrix.flatten('A')

	def encode_round_actions(self, round_actions):
		"""
		编码本轮动作序列
		"""
		if not round_actions:
			return np.zeros(36, dtype=np.float32)
		matrix = np.zeros((4, 9), dtype=np.float32)
		for idx, action in enumerate(round_actions):
			matrix[idx][self.action_id[action]] = 1

		return matrix.flatten('A')

	def encode_last_action(self, last_action):
		"""
		编码上一个动作
		"""
		if not last_action:
			return np.zeros(9, dtype=np.float32)
		matrix = np.zeros(9, dtype=np.float32)
		matrix[self.action_id[last_action]] = 1

		return matrix.flatten('F')

	def encode_legal_actions(self, actions):
		"""
		编码合法动作
		"""
		if not actions:
			return np.zeros(27, dtype=np.float32)

		matrix = np.zeros((3, 9), dtype=np.float32)
		for idx, action in enumerate(actions):
			matrix[idx][self.action_id[action]] = 1

		return matrix.flatten('A')

	@staticmethod
	def encode_cards(cards):
		"""
		卡牌编码
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards:
			return np.zeros(48, dtype=np.float32)
		matrix = np.zeros((4, 12), dtype=np.float32)
		cards_dict = Counter(cards)
		for card, num_times in cards_dict.items():
			matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1

		matrix = matrix.flatten('A')

		return matrix

	@staticmethod
	def compare_big_res(compare_big_cards):
		"""
		todo: 闲家撒扑时，比较庄闲大扑大小
		"""
		if not compare_big_cards or compare_big_cards == "draw":
			return np.zeros(18, dtype=np.float32)
		matrix = np.zeros((2, 9), dtype=np.float32)
		# 判断庄闲大小扑谁大，根据庄闲ID编码
		# 庄家大扑大
		if compare_big_cards == "landlord":
			matrix[0][:] = 1
		# 闲家大扑大
		elif compare_big_cards == "farmer":
			matrix[1][:] = 1
		return matrix.flatten('A')

	@staticmethod
	def encode_history_actions(history_actions):
		"""
		todo: 编码历史对局动作序列
		"""
		# 当前动作历史记录为空
		if not history_actions:
			return np.zeros(9 * 3, dtype=np.float32)

		# 提取动作选择取数范围
		if len(history_actions) > 9:
			start = len(history_actions) - 9
			end = len(history_actions)
			history_actions = history_actions[start:end]
		matrix = np.zeros((9, 3), dtype=np.float32)
		for idx, actions in enumerate(history_actions):
			for i, action in enumerate(actions):
				matrix[idx][i] = 1

		return matrix.flatten('A')

	@staticmethod
	def encode_big_and_small_cards(cards):
		"""
		编码大小扑卡牌
		"""
		# 闲家操作动作为密牌时，则对庄家隐藏卡牌信息，闲家编码全部为零
		if not cards:
			return np.zeros(48, dtype=np.float32)
		matrix = np.zeros((4, 12), dtype=np.float32)
		cards_dict = Counter(cards)
		for card, num_times in cards_dict.items():
			matrix[(card // 100) - 1][new_card_encoding_dict[card % 100]] = 1

		matrix = matrix.flatten('A')

		return matrix

	@staticmethod
	def encode_landlord_or_farmer(landlord_or_farmer, is_draw=False):
		"""
		判断当前玩家是否为庄家
		"""
		if not landlord_or_farmer or is_draw:
			return np.zeros(18, dtype=np.float32)
		matrix = np.zeros((2, 9), dtype=np.float32)
		if landlord_or_farmer == 'landlord':
			matrix[0][:] = 1
		else:
			matrix[1][:] = 1

		return matrix.flatten('A')

	def encode_history_round(self, history_rounds):
		"""
		编码历史对局
		"""
		# 对局信息记录为空时，则全部返回零
		if not history_rounds:
			return np.zeros(10 * 294, dtype=np.float32)
		# 十局历史记录编码
		collect_remain_by_matrix = np.zeros((10, 294), dtype=np.float32)
		if len(history_rounds) > 10:
			start = len(history_rounds) - 10
			end = len(history_rounds)
			history_rounds = history_rounds[start:end]
		for idx, history_round in enumerate(history_rounds):
			winner_id_round = self.encode_landlord_or_farmer(history_round["winner_role"], history_round["winner_id"] == -1)
			winner_big_cards = self.encode_big_and_small_cards(history_round["landlord_cards"][:2])
			winner_small_cards = self.encode_big_and_small_cards(history_round["landlord_cards"][2:])
			loser_big_cards = self.encode_big_and_small_cards(history_round["farmer_cards"][:2])
			loser_small_cards = self.encode_big_and_small_cards(history_round["farmer_cards"][2:])
			compare_winner_big_cards = self.compare_big_res(history_round["compare_big_cards"])
			compare_winner_small_cards = self.compare_big_res(history_round["compare_small_cards"])
			remain_by_matrix = self.encode_cards(history_round["remain_cards"])

			package_matrix = np.hstack((
				winner_id_round,
				winner_big_cards,
				winner_small_cards,
				loser_big_cards,
				loser_small_cards,
				compare_winner_big_cards,
				compare_winner_small_cards,
				remain_by_matrix,
			))

			collect_remain_by_matrix[idx] = package_matrix.flatten('A')

		return collect_remain_by_matrix.flatten('A')

sy_state = SyState()