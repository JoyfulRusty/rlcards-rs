# -*- coding: utf-8 -*-

import random
import functools

from rlcards.games.mahjong.utils import init_deck, lai_zi_deck, compare2cards_obj


class MahjongDealer:
	"""
	麻将发牌类
	"""
	def __init__(self, np_random):
		"""
		TODO: 初始化麻将发牌器属性参数
		"""
		self.np_random = np_random					# 随机种子
		self.deck = init_deck()						# 普通卡牌制作(3 x 9 x 4 = 108)
		self.magic_deck = lai_zi_deck()				# 万能卡牌制作(3 x 1 = 3)
		self.shuffle()								# 洗牌
		self.table = []								# 牌桌，存储玩家打出的牌
		self.curr_mo_card = 0						# 玩家当前摸牌
		self.left_count = 111	                    # 统计所有卡牌数量
		self.landlord_id = -1	                    # 开局庄家ID
		self.curr_mo_player = -1                   	# 当前摸牌玩家
		self.played_cards = [[], [], []]  			# 四位玩家的出牌记录
		self.record_action_seq_history = []         # 记录玩家出牌序列[[1], [2], ...]
		self.save_valid_operates = {}               # 存储有效操作

	def shuffle(self):
		"""
		洗牌，但不包含癞子牌
		"""
		self.np_random.shuffle(self.deck)

	def deal_cards(self, player, nums):
		"""
		TODO: 发牌 -> 12(卡牌) + 1癞子
		"""
		# 手牌(+12)
		for _ in range(nums):
			card = self.deck.pop()
			self.left_count -= 1
			player.curr_hand_cards.append(card.get_card_value())

		# 万能牌(+1)
		if self.magic_deck:
			magic_card = self.magic_deck.pop()
			self.left_count -= 1
			player.curr_hand_cards.append(magic_card.get_card_value())

		player.curr_hand_cards.sort(key=functools.cmp_to_key(compare2cards_obj))

	def mo_cards(self, player):
		"""
		TODO: 玩家摸牌
		"""
		# 摸牌(发一张牌)
		card = self.deck.pop()

		# 添加摸牌至手牌中
		self.left_count -= 1
		self.curr_mo_player = player
		self.curr_mo_card = card.get_card_value()

		player.curr_hand_cards.append(self.curr_mo_card)
		player.curr_hand_cards.sort(key=functools.cmp_to_key(compare2cards_obj))

	def play_card(self, player, card):
		"""
		TODO: 出牌
		"""
		# 删除玩家出牌
		card = player.curr_hand_cards.pop(player.curr_hand_cards.index(card))

		# 记录出牌历史
		self.table.append(card)
		self.played_cards[player.player_id].append(card)
		player.all_chu_cards.append((player.player_id, card))

	def set_landlord_id(self, players):
		"""
		TODO: 开局选庄
		"""
		player_id = [_ for _ in range(len(players))]

		# 设置庄家ID
		self.landlord_id = sorted(
			{p_id: random.randint(1, 6) for p_id in player_id}.items(),
			key=lambda x: x[1],
			reverse=True)[0][0]

		return self.landlord_id