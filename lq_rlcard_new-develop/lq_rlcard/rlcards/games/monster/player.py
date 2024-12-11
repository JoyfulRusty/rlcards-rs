# -*- coding: utf-8 -*-

from collections import Counter

from rlcards.const.monster.const import BASE_GOLD, CARD_VALUES


class MonsterPlayer:
	"""
	打妖怪游戏玩家
	"""
	def __init__(self, player_id, np_random):
		"""
		初始化打妖怪游戏中属性参数
		"""
		self.is_out = False  				 # 破产标识
		self.is_all = False                  # 手牌出完标识
		self.player_id = player_id           # 玩家ID
		self.np_random = np_random           # 随机数
		self.curr_hand_cards = []            # 手牌
		self.no_pass_pick_cards = []         # 捡牌
		self.pick_rewards = 0.0              # 捡牌动作奖励

	def pick_cards(self, players, picked_cards, golds, bust, curr_state):
		"""
		玩家捡牌
		"""
		# 存在合法动作，但选择主动捡牌
		if curr_state['actions']:
			# 存在合法牌，但是选择主动捡牌，则根据捡牌数量设置奖励
			self.card_rewards(picked_cards)
		# 添加捡牌
		self.no_pass_pick_cards.extend(picked_cards)

		# 计算捡牌之后金币数量
		# 计算本次捡牌所支付的金币及支付后手中的金币是否破产
		pay_after_golds = self.calc_pick_to_bust(players, golds, bust)

		return pay_after_golds

	def card_rewards(self, pick_cards):
		"""
		根据卡牌计算奖励
		"""
		for card in pick_cards:
			self.pick_rewards += CARD_VALUES.get(card % 100, 0)

	def calc_pick_to_bust(self, players, golds, bust):
		"""
		TODO: 计算玩家捡牌后，是否破产，破产玩家将退出本局游戏
		"""
		bust_flag = 0.0
		add_golds = []
		pay_pick_golds = len(self.no_pass_pick_cards) * BASE_GOLD
		# 无破产玩家
		if self.count_bust(bust) == 0:
			avg_picks_to_golds = float(pay_pick_golds / 3)
		# 有破产玩家
		else:
			# 统计破产玩家数量
			bust_nums = self.count_bust(bust)
			# 出现破产玩家之后，金币计算
			if bust_nums < 3:
				avg_picks_to_golds = float(pay_pick_golds / (3 - bust_nums))
			else:
				avg_picks_to_golds = float(pay_pick_golds)

		# 限制玩家金币扣除范围，当金币扣除大于手上金币时，金币最多扣除到-1600
		# 玩家向其他三位玩家赔付完当前捡牌需要支出的所有金币后手中最终的金币数量
		pay_after_golds = golds[self.player_id] - float(pay_pick_golds)
		golds[self.player_id] = pay_after_golds

		# 标记破产玩家
		if float(pay_after_golds) <= bust_flag:
			players[self.player_id].is_out = True

		# 找出当前符合支付金币的玩家
		for p_id, p in enumerate(players):
			if p_id == self.player_id:
				continue
			if p.is_out:
				continue
			else:
				add_golds.append((p_id, golds[p_id] + float(avg_picks_to_golds)))

		# TODO: 更新金币(捡牌动作)
		# 更新当前捡牌玩家金币
		golds[self.player_id] = pay_after_golds

		# 更新其他玩家金币
		for ag in add_golds:
			golds[ag[0]] = ag[1]

		return pay_after_golds

	def count_bust(self, bust):
		"""
		统计破产玩家数量
		"""
		count = 0
		for pos, flag in bust.items():
			if not flag:
				continue
			count += 1
		return count