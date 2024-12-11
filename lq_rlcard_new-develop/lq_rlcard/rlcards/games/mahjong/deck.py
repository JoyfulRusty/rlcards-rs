# -*- coding: utf-8 -*-

import random

from collections import Counter


class DeckCards:
	"""
	设置好牌:
		1.开局发好牌
		2.游戏对局中发好牌
	"""
	# 定义一个变量__slots__，它的作用是阻止在实例化类时为实例分配dict
	# 默认情况下每个类都会有一个dict, 通过__dict__访问，这个dict维护了这个实例的所有属性
	__slots__ = [
		"all_cards",
		"deal_cards",
		"curr_card",
		"player_nums",
		"sample_count",
		"remain_cards_count"
	]

	def __init__(self):
		"""
		初始化卡牌属性参数
		"""
		self.all_cards = []
		self.deal_cards = []
		self.curr_card = 0
		self.player_nums = 0
		self.sample_count = 13
		self.remain_cards_count = {}

	def update_attr(self, all_cards, player_nums):
		"""
		更新当前卡牌属性
		"""
		self.all_cards = all_cards
		self.player_nums = player_nums
		self.sample_count = self.sample_count * player_nums

	def sample_cards(self):
		"""
		采样对应卡牌
		"""
		# 发牌
		self.deal_cards = random.sample(self.all_cards, k=self.sample_count)
		# 剩余卡牌
		self.remain_cards_count = self.remain_cards()

	def set_cards(self):
		"""
		给每位玩家发牌13张
		"""
		self.sample_cards()
		random.shuffle(self.deal_cards)
		random.shuffle(self.deal_cards)
		cards_len = len(self.deal_cards) // self.player_nums
		for i in range(self.player_nums):
			curr_hand_cards = self.deal_cards[i * cards_len: (i + 1) * cards_len]
			print(sorted(curr_hand_cards))

		self.deal_cards = []

	def remain_cards(self):
		"""
		计算开局发牌之后，剩余的卡牌
		"""
		one_list = []
		two_list = []
		three_list = []
		four_list = []
		for card in list(set(self.all_cards)):
			if card in self.deal_cards:
				count = self.deal_cards.count(card)
				for _ in range(count):
					self.all_cards.remove(card)

		count_remain_cards = Counter(self.all_cards)
		for card, count in count_remain_cards.items():
			if count == 1:
				one_list.append(card)
			elif count == 2:
				two_list.append(card)
			elif count == 3:
				three_list.append(card)
			else:
				four_list.append(card)

		return one_list, two_list, three_list, four_list

	def game_start(self, all_cards, player_id):
		"""
		TODO: 游戏开局时，向对应ID(发好牌请求玩家ID)，发牌

		游戏开始时，向指定座位号发好牌
		"""
		pass

	def game_mo_cards(self):
		"""
		TODO: 游戏摸牌时，向对应ID(发好牌请求玩家ID)，发牌
		1.平胡发好牌
		2.大对子发好牌
		3.小七对发好牌
		4.龙七对发好牌
		5.清一色发好牌

		根据当前手牌情况，发对应牌型好牌
		或考虑通过对比向听数来进行发好牌
		"""
		pass


ALL_CARDS = [
	11, 12, 13, 14, 15, 16, 17, 18, 19,  # 万
	21, 22, 23, 24, 25, 26, 27, 28, 29,  # 条
	31, 32, 33, 34, 35, 36, 37, 38, 39,  # 筒
	]

build_108 = sorted(ALL_CARDS * 4)

def main(all_cards):
	"""
	主程序
	"""
	deck = DeckCards()
	deck.update_attr(all_cards, 3)
	deck.set_cards()

if __name__ == "__main__":
	main(build_108)
