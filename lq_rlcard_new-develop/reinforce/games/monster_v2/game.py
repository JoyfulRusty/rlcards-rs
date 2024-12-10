# -*- coding: utf-8 -*-

import copy
import torch
import random

from reinforce.share.base_game import BaseGame
from reinforce.games.monster_v2.rule import Rule
from reinforce.games.monster_v2.poker import MonsterPoker
from reinforce.games.monster_v2.state import MonsterState
from reinforce.games.monster_v2.card import MonsterCards as Cards
from reinforce.games.monster_v2.player import MonsterPlayer as Player


class MonsterGame(BaseGame):
	"""
	Class for the Monster Game
	"""

	def __init__(self):
		"""
		Initializes the game
		"""
		super().__init__()
		self.bust_infos = {}
		self.num_players = 4
		self.base_golds = 100
		self.episode_return = {}
		self.init_gold_infos = {}
		self.is_played_shi_fu = False  # is played shi fu ?
		self.position_maps = {0: "down", 1: "right", 2: "up", 3: "left"}

	def reset(self):
		"""
		Resets the game
		"""
		return self.init_game()

	def init_game(self):
		"""
		Initializes the game
		"""
		# Clear the game state parameters
		self.clear()
		self.traces = []
		self.round_cards = []
		self.state = MonsterState()
		self.pokers = MonsterPoker()
		self.remain_cards = self.pokers.all_cards
		self.players = [Player(i) for i in range(self.num_players)]
		self.deal_cards()
		self.init_golds()
		self.is_played_shi_fu = False
		self.curr_seat_id = self.landlord_id
		self.curr_p = self.players[self.curr_seat_id]
		self.bust_infos = {self.position_maps.get(curr_p.seat_id): curr_p.is_out for curr_p in self.players}
		self.episode_return = {self.position_maps.get(curr_p.seat_id): torch.zeros(1, 1) for curr_p in self.players}
		return self.get_state()

	def init_golds(self):
		"""
		Initializes the golds
		"""
		for curr_p in self.players:
			curr_p.golds = round(random.uniform(a=1600.0, b=3200.0), 2)
		init_gold_infos = {curr_p.seat_id: curr_p.golds for curr_p in self.players}
		self.init_gold_infos = copy.deepcopy(init_gold_infos)

	def deal_cards(self):
		"""
		Deals the cards to the players
		"""
		all_cards = self.pokers.deal_cards(player_nums=4, card_nums=7, extra_nums=1)
		for curr_p in self.players:
			curr_p.hand_cards = all_cards[curr_p.seat_id]
			if curr_p.hand_cards.count(Cards.FK_J):
				self.landlord_id = curr_p.seat_id

	def step(self, action):
		"""
		Steps the game
		"""
		# Is game over?
		done = False
		# Record the last action
		self.last_action = action
		# Get action value
		action_value = self.last_action.val
		# Record action history
		self.action_history.append([self.last_action])
		# If last action != pick action and last action == Universal action
		action_value = self.update_universal_map_val()
		turn = [self.curr_p.seat_id, self.last_action, action_value]
		self.traces.append(turn)
		# print(f"输出历史动作记录: ", self.traces)
		# Judge is played shi fu ?
		if not self.is_played_shi_fu and self.last_action.val == Rule.SHI_FU:
			self.is_played_shi_fu = True
		# Judge is played or Picked
		if self.last_action != Cards.PICK_CARD:
			self.round_cards.append(turn)
			# Current Player played cards
			self.curr_p.play_cards(self.last_action)
			if self.last_action in self.remain_cards:
				self.remain_cards.remove(self.last_action)
			if not self.curr_p.hand_cards:
				self.curr_p.is_all = True
			self.get_next_seat_id()
		else:
			# Step next action and next player
			round_cards = self.get_round_cards()
			self.curr_p.pick_cards(round_cards)
			# Calculate is Bust
			self.calc_pick_to_golds()
			self.round_cards = []
			# Is bust ?
			if self.curr_p.is_out:
				for card in self.curr_p.hand_cards:
					if card in self.remain_cards:
						self.remain_cards.remove(card)
				# If is out, look for next player
				self.get_next_seat_id()
			# Not bust!
			else:
				self.last_seat_id = self.curr_p.seat_id
				self.curr_seat_id = self.last_seat_id
				self.curr_p = self.players[self.curr_seat_id]

		# Is game over?
		if self.is_game_over():
			# print(f"Is Game Over: {self.is_game_over()}")
			# Game over
			done = True
			# Get winner
			self.winner_id = self.calc_winner_id()
			# Update the state rewards
			self.episode_return = self.get_payoffs()
			return {}, self.episode_return, done

		# Update the model state
		return self.get_state(), self.episode_return, done

	def update_universal_map_val(self):
		"""
		Update the universal map value
		"""
		action_value = self.last_action.val
		if self.last_action == Cards.UNIVERSAL_CARD:
			# Judge universal card is first play
			if not self.round_cards:
				action_value = Rule.XIAO_YAO
			else:
				last_action_value = self.traces[-1][-1]
				if Rule.is_monster(last_action_value):
					action_value = Rule.SHA_SENG
				elif Rule.is_tu_di(last_action_value):
					action_value = Rule.SHI_FU
				else:
					action_value = Rule.XIAO_YAO
		return action_value

	def get_next_seat_id(self):
		"""
		Gets the next seat id
		"""
		next_seat_id = None
		if self.curr_seat_id != 3:
			for next_id in range(self.curr_seat_id + 1, len(self.players)):
				if self.players[next_id].is_out or self.players[next_id].is_all:
					continue
				next_seat_id = next_id
				break
			if next_seat_id is None:
				for next_id in range(0, self.curr_seat_id):
					if not next_seat_id and self.players[next_id].is_out:
						continue
					if self.players[next_id].is_out or self.players[next_id].is_all:
						continue
					next_seat_id = next_id
					break
		else:
			for next_id in range(0, self.curr_seat_id):
				if self.players[next_id].is_out or self.players[next_id].is_all:
					continue
				next_seat_id = next_id
				break
		if next_seat_id is None:
			next_seat_id = self.curr_seat_id
		# Update Curr Player Step State Infos
		self.last_seat_id = self.curr_seat_id
		self.curr_seat_id = next_seat_id
		self.curr_p = self.players[self.curr_seat_id]

	def get_state(self):
		"""
		Gets the state of the game
		"""
		self.state.init_attrs(
			seat_id=self.curr_p.seat_id,
			hand_cards=self.curr_p.hand_cards,
			played_cards=self.curr_p.played_cards,
			last_action=self.get_last_action(),
			legal_actions=self.get_legal_actions(),
			round_cards=self.get_round_cards(),
			remain_cards=self.remain_cards,
			action_history=self.action_history,
			other_played_cards=self.get_other_played_cards(),
			other_left_cards=self.get_other_left_cards(),
			bust_infos=self.bust_infos,
		)
		return self.state.get_obs()

	def get_legal_actions(self):
		"""
		Gets the legal actions
		"""
		if not self.traces:
			return [Cards.FK_J]
		elif not self.round_cards:
			return self.curr_p.hand_cards
		else:
			can_play_cards = Rule.legal_cards(self.round_cards, self.curr_p.hand_cards)
			if self.is_played_shi_fu:
				can_play_cards.append(Cards.PICK_CARD)
			if not can_play_cards:
				can_play_cards.append(Cards.PICK_CARD)
			return can_play_cards

	def get_round_cards(self):
		"""
		Gets the round cards
		"""
		round_cards = [cards[1] for cards in self.round_cards]
		return round_cards

	def get_last_action(self):
		"""
		Gets the last 2 actions
		"""
		if not self.traces:
			return []
		if len(self.traces) < 4:
			return [trace[1] for trace in self.traces]
		return [trace[1] for trace in self.traces[-3:]]

	def get_other_left_cards(self):
		"""
		Gets the other hand cards
		"""
		return {
			self.position_maps.get(other_p.seat_id): len(other_p.hand_cards) for other_p in self.players
			if other_p != self.curr_p
		}

	def get_other_played_cards(self):
		"""
		Gets the other played cards
		"""
		return {
			self.position_maps.get(other_p.seat_id): other_p.played_cards for other_p in self.players
			if other_p != self.curr_p
		}

	def get_payoffs(self):
		"""
		Gets the payoffs
		"""
		# Judge is game over?
		if not self.is_game_over():
			return self.episode_return

		# Calculate game over did not pick cards
		pick_infos = [trace[-1] for trace in self.traces if trace[-1] == Cards.PICK_CARD.val]
		if not pick_infos:
			for eps in self.episode_return:
				self.episode_return[eps] = torch.as_tensor(1.0).view(1, 1)
			return self.episode_return

		# Calculate game over did pick cards
		winner = self.players[self.winner_id]
		self.episode_return[self.position_maps.get(winner.seat_id)] = torch.as_tensor(0.32).view(1, 1)
		for other_p in self.players:
			reward = 0.0
			if not other_p.picked_cards:
				reward += 1.0
			if other_p.is_out:
				reward -= 1.0
			reward += self.init_gold_infos.get(other_p.seat_id) * 0.0001
			self.episode_return[self.position_maps.get(other_p.seat_id)] = torch.as_tensor(reward).view(1, 1)
		# print("Episode Return:", self.episode_return)
		return self.episode_return

	def is_game_over(self):
		"""
		Checks if the game is over
		"""
		over_played = [other_p for other_p in self.players if other_p.is_out or other_p.is_all]
		return True if len(over_played) == 4 else False

	def calc_winner_id(self):
		"""
		Calculates the winner id
		"""
		all_golds = {other_p.seat_id: other_p.golds for other_p in self.players}
		self.init_gold_infos = {seat_id: all_golds[seat_id] - golds for seat_id, golds in self.init_gold_infos.items()}
		winner_id, max_golds = max(self.init_gold_infos.items(), key=lambda x: x[1])
		return winner_id

	def calc_pick_to_golds(self):
		"""
		Calculates the mini golds
		"""
		pick_len = len(self.round_cards)
		online_player, online_nums = self.calc_online_infos()
		pay_golds = (-pick_len * online_nums) * self.base_golds
		if self.curr_p.golds + pay_golds < 0:
			# Record is out no hand cards
			self.curr_p.is_out = True
			self.curr_p.is_all = True
			self.curr_p.hand_cards = []
		# Curr player finally Golds
		self.curr_p.golds = self.curr_p.golds + pay_golds
		# Paying Golds to each player
		each_pay_golds = len(self.round_cards) * self.base_golds
		for online_p in online_player:
			online_p.golds += each_pay_golds

	def calc_online_infos(self):
		"""
		Calculates the online nums
		"""
		online_players = [
			other_p for other_p in self.players
			if other_p != self.curr_p and (not other_p.is_out or not other_p.is_all)
		]
		return online_players, len(online_players)