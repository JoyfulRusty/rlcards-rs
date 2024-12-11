# -*- coding: utf-8 -*-

import functools

from rlcards.games.doudizhu.utils import cards2str, ddz_sort_card_obj, get_gt_card

class DdzPlayer:
	"""
	斗地主玩家对象
	存储玩家手中的卡牌和角色
	根据规则决定可以做出的动作，并执行相应的动作
	"""
	def __init__(self, player_id, np_random):
		"""
		玩家属性初始化
		:param player_id: 玩家id
		:param np_random: 随机数
		:param role: 玩家的临时角色(地主/农民)
		:param initial_hand: 初始卡牌
		:param _current_hand: 玩家手牌
		"""
		self.np_random = np_random
		self.player_id = player_id
		self.initial_hand = None
		self._current_hand = []
		self.role = ''
		self.played_cards = None
		self.singles = '3456789TJQKA2BR'
		# 记录每次play()从self._current_hand移除的牌
		# 并在play_back()时将牌恢复到self._current_hand
		self._record_played_cards = []

	@property
	def current_hand(self):
		return self._current_hand

	def set_current_hand(self, value):
		self._current_hand = value

	def get_state(self, public, others_hands, num_cards_left, actions):
		"""
		获取状态
		:param public:
		:param others_hands:
		:param num_cards_left:
		:param actions:
		:return:
		"""
		state = {}
		state['landlord'] = public['landlord']
		state['trace'] = public['trace'].copy()  # 桌牌
		state['played_cards'] = public['played_cards']
		state['self'] = self.player_id
		state['current_hand'] = cards2str(self._current_hand)
		state['others_hand'] = others_hands
		state['num_cards_left'] = num_cards_left
		state['actions'] = actions

		return state

	def available_actions(self, greater_player=None, judge=None):
		"""
		获取可以根据规则进行的操作
		:param greater_player(obj): 当前最大牌的玩家
		:param judge: 斗地主判别器对象
		:return: list: 动作串列表. Eg: ['pass', '8', '9', 'T', 'J']
		"""
		actions = []  # 玩家要不起的时候会添加pass，说明玩家要不起
		if greater_player is None or greater_player.player_id == self.player_id:
			actions = judge.get_playable_cards(self)
		else:
			actions = get_gt_card(self, greater_player)
		return actions

	def play(self, action, greater_player=None):
		"""
		执行动作
		:param action(string): 具体动作
		:param greater_player(obj): 打出当前最大牌的玩家
		:return: 如果有新的greater_player，返回它，如果没有，返回None
		"""
		trans = {'B': 'BJ', 'R': 'RJ'}
		if action == 'pass':
			self._record_played_cards.append([])
			return greater_player
		else:
			removed_cards = []
			self.played_cards = action
			for play_card in action:
				if play_card in trans:
					play_card = trans[play_card]
				for _, remain_card in enumerate(self._current_hand):
					if remain_card.rank != '':
						remain_card = remain_card.rank
					else:
						remain_card = remain_card.suit
					if play_card == remain_card:
						removed_cards.append(self.current_hand[_])
						self._current_hand.remove(self._current_hand[_])
						break
			self._record_played_cards.append(removed_cards)
			return self

	def play_back(self):
		"""
		将卡牌恢复到self._current_hand
		"""
		removed_cards = self._record_played_cards.pop()
		self._current_hand.extend(removed_cards)
		self._current_hand.sort(key=functools.cmp_to_key(ddz_sort_card_obj))



