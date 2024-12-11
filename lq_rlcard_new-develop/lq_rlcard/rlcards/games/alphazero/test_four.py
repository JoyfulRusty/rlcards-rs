# -*- coding: utf-8 -*-

from unittest import TestCase
from rlcards.games.alphazero.connect_four_game import ConnectFourGame


class TestConnectFourGame(TestCase):
	def test_check_game_over1(self):
		game = ConnectFourGame()
		game.state = [
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 1, -1, 0],
			[0, 0, 0, 1, 1, -1, 0],
			[0, 0, 1, -1, -1, -1, 0]
		]

		game_over, value = game.check_game_over(1)
		self.assertEqual(game_over, True)
		self.assertEqual(value, 1)

	def test_check_game_ove2(self):
		game = ConnectFourGame()
		game.state = [
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, -1, 0],
			[0, 0, 0, 0, 1, -1, 0],
			[0, 0, 0, 1, 1, -1, 0],
			[0, 0, 1, 1, 1, -1, 0]
		]

		game_over, value = game.check_game_over(1)
		self.assertEqual(game_over, True)
		self.assertEqual(value, -1)

	def test_check_game_ove3(self):
		game = ConnectFourGame()
		game.state = [
			[1, -1, 1, -1, 1, -1, 1],
			[1, 1, -1, -1, 1, 1, -1],
			[-1, 1, -1, -1, -1, 1, -1],
			[-1, 1, -1, 1, 1, -1, 1],
			[-1, -1, 1, 1, 1, -1, 1],
			[1, -1, -1, 1, -1, 1, -1]
		]

		game_over, value = game.check_game_over(1)
		self.assertEqual(game_over, True)
		self.assertEqual(value, 0)

	def test_check_game_over4(self):
		game = ConnectFourGame()
		game.state = [
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, -1, 0, 0, 0, 0],
			[0, 0, 1, 0, -1, 0, 0],
			[1, 0, -1, 1, -1, 0, 0]
		]

		game_over, value = game.check_game_over(1)
		self.assertEqual(game_over, False)
		self.assertEqual(value, 0)