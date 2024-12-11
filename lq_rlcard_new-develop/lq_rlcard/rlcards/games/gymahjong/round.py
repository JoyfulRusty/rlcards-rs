# -*- coding: utf-8 -*-

from rlcards.const.mahjong import const
from rlcards.utils.mahjong.xxc import RuleXXC
from rlcards.const.mahjong.const import ActionType, ACTION_PRIORITY, JiType_FK, JiType_XL, OverType, CardType

from rlcards.games.mahjong.bird import \
    other_ming_xi_data, \
    self_ming_xi_data, \
    update_result_score, \
    card_count_fk


class MahjongRound:
	"""
	麻将游戏流程
	"""
	def __init__(self, judge, dealer, num_players, np_random):
		"""
		初始化round属性参数
		"""
		self.judge = judge
		self.dealer = dealer
		self.num_players = num_players          # 玩家数量
		self.np_random = np_random              # 生成随机数
		self.last_cards = 0                     # 上一位玩家的出牌
		self.curr_player_id = -1  				# 当前玩家ID
		self.last_player_id = -1  				# 上一位玩家ID
		self.valid_actions = False              # 玩家有效的合法动作
		self.bird_count = 1  					# 翻鸡统计
		self.shang_xia_ji = 0  					# 上下鸡
		self.base_ji_bei_lv = 1  				# 翻鸡基础倍数
		self.__fk_fan_ji_pai = 1                # 房卡翻鸡牌
		self.__fk_ben_ji = 0                    # 本鸡
		self.__fk_cfj = 1                       # 冲锋鸡(房卡)
		self.__fk_yao_bai_ji = 0                # 摇摆鸡
		self.__fk_ze_ren_ji = 1                 # 房卡责任鸡
		self.__fk_ji_pai = None                 # 房卡鸡牌
		self.__fk_man_tang_ji = 0               # 满堂鸡
		self.__fk_wu_gu_ji = 0                  # 房卡乌骨鸡
		self.__fk_default_ji = {CardType.YAO_JI}  # 房卡默认鸡
		if self.__fk_wu_gu_ji:
			self.__fk_default_ji.add(CardType.WU_GU_JI)
		self.__fk_bao_ji = 1                    # 房卡包鸡
		self.__fk_zj = 0                        # 房卡默认站鸡
		self.__round_first_ji = 0  				# 记录冲锋鸡
		self.__round_first_wgj = 0  			# 记录冲锋乌骨鸡
		self.__ze_ren_ji_seat_id = 0  			# 责任鸡赔付玩家
		self.__ze_ren_ji_win_seat_id = 0  		# 责任鸡得到玩家
		self.__ze_ren_wgj_seat_id = 0  			# 责任鸡玩家
		self.__ze_ren_wgj_win_seat_id = 0       # 责任乌骨鸡得到玩家
		self.__cfj_seat_id = 0  				# 冲锋幺鸡鸡玩家
		self.__cf_wgj_seat_id = 0  				# 冲锋乌骨鸡ID
		self.__record_fan_ji_pai = {}  			# 记录桌子内的鸡牌，只要有人胡牌，就有一张鸡牌
		self.__player_actions = []  			# 出牌时，存储其他玩家的动作，选取最大优先级别进行出牌

	def proceed_round(self, players, action):
		"""
		TODO: ===== 判断玩家操作流程 =====
		"""
		# 添加所有打牌记录，包括(碰、杠、胡)
		self.dealer.record_action_seq_history.append(action)
		curr_player = self.calc_curr_player(players, self.curr_player_id)

		# TODO: 打牌 -> 判断是否要牌 -> 摸牌
		if isinstance(action, int):
			self.dealer.play_card(curr_player, action)
			self.last_player_id = curr_player.player_id
			self.action_check_call(players, curr_player)
		else:
			# TODO: save_valid_operates中选择最高优先级别作为首出玩家
			# 	1.胡(捡/开)
			#   2.杠(明杠)
			#   3.碰
			#   4.过

			# 判断是否存在多个玩家操作当前动作不为密胡时，则进入此流程，密胡直接进行密操作即可
			# 胡开提前判断，选择胡开则游戏结束，不进入此流程
			if len(self.dealer.save_valid_operates) > 1 and action != ActionType.MI_HU:
				# 删除当前已经选择操作的玩家记录
				self.dealer.save_valid_operates.pop(curr_player.player_id)
				# 当前动作为捡胡时，直接进行捡胡操作
				if action == ActionType.PICK_HU:
					# 直接进行捡胡
					self.action_pick_hu(players, curr_player)

				# 动作为过时，可进行碰/明杠/捡胡
				# 动作为捡胡时，不能碰或明杠
				# 判断当前玩家选择过时，其他玩家是否选择操作
				if action == ActionType.G_PASS:
					# 计算剩余未操作的玩家和有效操作
					p_id = list(self.dealer.save_valid_operates.keys())[0]
					valid_actions = list(self.dealer.save_valid_operates.values())[0]
					self.curr_player_id = p_id
					self.valid_actions = valid_actions

			# 计算索杠 -> 转弯杠
			elif action == ActionType.SUO_GONG:
				# 判断其他玩家是否能胡
				curr_player.suo_gang_state = True  # 设置玩家处于杠牌状态中，用于计算杀杠
				self.dealer.save_valid_operates = {}
				valid_actions = [ActionType.G_PASS]
				for player in players:
					if player.player_id == curr_player.player_id:
						continue
					can_hu, _ = RuleXXC.get_hu_type(
						player,
						players,
						curr_player.player_id,
						self.dealer.landlord_id,
						self.dealer.curr_mo_card,
						False
					)
					# 没胡则继续判断下一位玩家
					if not can_hu:
						continue
					# 添加玩家胡牌操作
					else:
						valid_actions.append(ActionType.KAI_HU)
						self.dealer.save_valid_operates[player.player_id] = valid_actions
				# 此处，靠前玩家优先操作
				if not self.dealer.save_valid_operates:
					for p_id, extra_valid_actions in self.dealer.save_valid_operates.items():
						self.last_player_id = curr_player.player_id
						self.curr_player_id = p_id
						self.valid_actions = extra_valid_actions
						break
				# 执行索杠操作
				else:
					self.action_suo_gang(players, curr_player, action)

			else:
				self.enter_do_actions(curr_player, players, action)

	def enter_do_actions(self, curr_player, players, action):
		"""
		TODO: 执行具体操作
		"""
		# TODO: 过
		if action == ActionType.G_PASS:
			self.action_pass(players, curr_player)

		# TODO: 报听
		elif action == ActionType.TIAN_TING:
			self.action_tian_ting(curr_player, action)

		# TODO: 碰
		elif action == ActionType.PONG:
			self.action_pong(players, curr_player, action)

		# TODO: 明杠
		elif action == ActionType.MING_GONG:
			self.action_ming_gang(players, curr_player, action)

		# TODO: 鞍钢
		elif action == ActionType.AN_GONG:
			self.action_an_gang(players, curr_player, action)

		# TODO: 密胡
		elif action == ActionType.MI_HU:
			self.action_mi_hu(players, curr_player)

		# TODO: 捡胡
		elif action == ActionType.PICK_HU:
			self.action_pick_hu(players, curr_player)

	@staticmethod
	def calc_curr_player(players, player_id):
		"""
		计算当前玩家
		"""
		return players[player_id]

	def action_pass(self, players, curr_player):
		"""
		玩家选择过
		"""
		# 分析: 两种情况
		# 摸牌: 需要出牌
		# 更换下一位操作玩家ID
		self.curr_player_id = (self.last_player_id + 1) % 3
		self.last_player_id = curr_player.player_id

		# 下一位操作玩家
		next_player = self.calc_curr_player(players, self.curr_player_id)

		# 补牌
		self.dealer.mo_cards(next_player)

		# 是否存在有效动作
		valid_actions = self.judge.calc_operates_after_mo_pai(next_player, players, self.dealer, self.last_player_id)

		if valid_actions:
			self.valid_actions = valid_actions
		else:
			self.valid_actions = False

	def action_tian_ting(self, curr_player, action):
		"""
		玩家选择报听
		"""
		# 天听
		curr_player.tian_ting = True
		curr_player.can_tian_ting = False
		curr_player.rewards += 0.2

		self.judge.do_tian_ting(curr_player)

		# 天听数据
		ting_card_infos = {
			"action": action,
			"card": curr_player.ting_list,
			"id": curr_player.player_id
		}
		curr_player.ting_infos.append(ting_card_infos)

		self.last_player_id = curr_player.player_id
		self.curr_player_id = curr_player.player_id
		self.valid_actions = False

	def action_pong(self, players, curr_player, action):
		"""
		玩家选择碰牌
		"""
		# 碰牌
		self.judge.do_pong(curr_player, self.last_cards)

		# 碰杠数据
		pong_result_infos = {
			"action": action,
			"pong_card": self.last_cards,
			"player_id": curr_player.player_id
		}
		curr_player.pong_infos.append(pong_result_infos)

		# # TODO: 计算责任鸡(血流红中)
		# ze_ren_ji_result =  self.deal_ze_ren_ji(players)
		# if ze_ren_ji_result:
		# 	if ze_ren_ji_result == 1:
		# 		# 责任鸡
		# 		pong_result_infos["ze_ren_ji"] = self.__ze_ren_ji_seat_id
		# 	else:
		# 		# 责任乌骨鸡
		# 		pong_result_infos["ze_ren_ji"] = self.__ze_ren_wgj_seat_id

		# 更新ID
		self.last_player_id = curr_player.player_id
		self.curr_player_id = curr_player.player_id
		self.valid_actions = False

	def action_ming_gang(self, players, curr_player, action):
		"""
		玩家选择明杠
		"""
		self.judge.do_ming_gang(curr_player, self.dealer.table[-1])

		# 动作奖励
		curr_player.rewards += 0.1
		players[self.last_player_id].rewards -= 0.1

		# 杠牌数据
		ming_gang_result = {
			"action": action,
			"gang_card": self.dealer.table[-1],
			"player_id": curr_player.player_id
		}
		curr_player.gang_infos.append(ming_gang_result)

		# # TODO: 计算责任鸡(血流红中)
		# ze_ren_ji_result = self.deal_ze_ren_ji(players)
		# if ze_ren_ji_result:
		# 	if ze_ren_ji_result == 1:
		# 		# 责任鸡玩家ID
		# 		ming_gang_result["ze_ren_ji"] = self.__ze_ren_ji_seat_id
		# 	else:
		# 		# 责任乌骨鸡玩家ID
		# 		ming_gang_result["ze_ren_ji"] = self.__ze_ren_wgj_seat_id

		self.dealer.mo_cards(curr_player)

		# 杠后摸牌信息
		mo_card = self.dealer.curr_mo_card
		mo_card_infos = {
			"gang_hou_mo_card": mo_card,
			"player_id": curr_player.player_id,
			"action": ActionType.MING_GONG
		}
		curr_player.gang_hou_mo_card.append(mo_card_infos)

		# # 更新ID
		self.last_player_id = curr_player.player_id
		self.curr_player_id = curr_player.player_id

		valid_actions = self.judge.calc_operates_after_mo_pai(curr_player, players, self.dealer, self.last_player_id)

		if valid_actions:
			self.valid_actions = valid_actions
		else:
			self.valid_actions = False

	def action_suo_gang(self, players, curr_player, action):
		"""
		玩家选择索杠
		"""
		# 删除杠牌(索杠之前，为暗杠，仅删除1张)
		self.judge.do_suo_gang(curr_player, self.dealer.curr_mo_card)

		# 动作奖励
		curr_player.rewards += 0.12
		for player in players:
			if player.player_id == curr_player.player_id:
				continue
			player.rewards -= 0.12

		# 杠牌数据
		suo_gang_result = {
			"action": action,
			"gang_card": self.dealer.curr_mo_card,
			"player_id": curr_player.player_id
		}
		curr_player.gang_infos.append(suo_gang_result)

		self.dealer.mo_cards(curr_player)

		# 杠后摸牌信息
		mo_card = self.dealer.curr_mo_card
		mo_card_infos = {
			"gang_hou_mo_card": mo_card,
			"player_id": curr_player.player_id,
			"action": ActionType.SUO_GONG
		}
		curr_player.gang_hou_mo_card.append(mo_card_infos)
		# 更新玩家ID
		self.last_player_id = curr_player.player_id
		self.curr_player_id = curr_player.player_id

		valid_actions = self.judge.calc_operates_after_mo_pai(curr_player, players, self.dealer, self.last_player_id)

		if valid_actions:
			self.valid_actions = valid_actions
		else:
			self.valid_actions = False

	def action_an_gang(self, players, curr_player, action):
		"""
		玩家选择暗杠
		"""
		# 删除杠牌
		self.judge.do_an_gang(curr_player, self.dealer)

		# 动作奖励
		curr_player.rewards += 0.15
		for player in players:
			if player.player_id == curr_player.player_id:
				continue
			player.rewards -= 0.15

		# 杠牌数据
		an_gang_result = {
			"action": action,
			"gang_card": self.dealer.curr_mo_card,
			"player_id": curr_player.player_id
		}
		curr_player.gang_infos.append(an_gang_result)

		self.dealer.mo_cards(curr_player)

		# 杠后摸牌信息
		mo_card = self.dealer.curr_mo_card
		mo_card_infos = {
			"gang_hou_mo_card": mo_card,
			"player_id": curr_player.player_id,
			"action": ActionType.AN_GONG
		}
		curr_player.gang_hou_mo_card.append(mo_card_infos)

		# 更新玩家ID
		self.last_player_id = curr_player.player_id
		self.curr_player_id = curr_player.player_id

		valid_actions = self.judge.calc_operates_after_mo_pai(curr_player, players, self.dealer, self.last_player_id)

		if valid_actions:
			self.valid_actions = valid_actions
		else:
			self.valid_actions = False

	def action_mi_hu(self, players, curr_player):
		"""
		玩家选择密胡
		"""
		# 判断当前密胡操作是否为天胡
		if curr_player.player_id == self.dealer.landlord_id and not curr_player.all_chu_cards:
			curr_player.tian_hu = True

		# 密胡翻鸡
		# self.do_fan_bird_xl_hz()

		# 胡牌奖励
		curr_player.rewards += 0.25
		for player in players:
			if player.player_id == curr_player.player_id:
				continue
			player.rewards -= 0.25

		# 必闷一次
		curr_player.bi_men_yi_shou = True
		self.judge.mi_hu(curr_player, players, self.dealer, self.last_player_id)

		self.last_player_id = curr_player.player_id
		self.curr_player_id = curr_player.player_id
		self.valid_actions = False

	def action_pick_hu(self, players, curr_player):
		"""
		玩家选择捡胡
		"""
		# 玩家捡，翻鸡
		# self.do_fan_bird_xl_hz()

		# 动作奖励
		players[self.last_player_id].rewards -= 0.25
		curr_player.rewards += 0.25

		self.judge.pick_hu(curr_player, players, self.dealer, self.last_player_id)
		self.last_player_id = self.last_player_id
		self.curr_player_id = (self.last_player_id + 1) % 3
		next_player = players[self.curr_player_id]

		# 摸牌
		self.dealer.mo_cards(next_player)

		valid_actions = self.judge.calc_operates_after_mo_pai(next_player, players, self.dealer, self.last_player_id)

		if valid_actions:
			self.valid_actions = valid_actions
		else:
			self.valid_actions = False

	def action_check_call(self, players, curr_player):
		"""
		检查出牌和摸牌流程操作
		"""
		# 计算玩家杠后出牌
		if curr_player.player_id == self.last_player_id and len(self.dealer.record_action_seq_history) > 1:
			if self.dealer.record_action_seq_history[-2] in [ActionType.MING_GONG, ActionType.SUO_GONG, ActionType.AN_GONG]:
				# 杠后出牌数据
				gang_hou_infos = {
					"gang_hou_chu_card": self.dealer.record_action_seq_history[-1],
					"player_id": curr_player.player_id,
					"action": self.dealer.record_action_seq_history[-2]
				}
				# 添加杠后出牌信息
				curr_player.gang_hou_chu_card.append(gang_hou_infos)

		# 清除上一轮记录的有效操作
		self.dealer.save_valid_operates = {}
		# 判断出牌后，其他玩家操作
		for player in players:
			if player == curr_player:
				continue
			# 计算其他玩家操作(判断其他玩家是否能够进行其他操作 -> 碰、杠、胡)
			valid_actions = self.judge.calc_operates_after_chu_pai(player, players, self.dealer, self.last_player_id)
			# 条件符合则打上标签
			if len(valid_actions) > 1:
				self.dealer.save_valid_operates[player.player_id] = valid_actions

		# TODO: 仅有一位玩家有操作
		if len(self.dealer.save_valid_operates) == 1:
			self.last_cards = self.dealer.table[-1]
			self.last_player_id = curr_player.player_id
			self.curr_player_id = list(self.dealer.save_valid_operates.keys())[0]
			self.valid_actions = self.dealer.save_valid_operates[self.curr_player_id]

		# TODO: 其他两位玩家都有操作
		elif len(self.dealer.save_valid_operates) > 1:
			self.last_cards = self.dealer.table[-1]
			self.last_player_id = curr_player.player_id
			self.curr_player_id, self.valid_actions = self.max_priority_action(self.dealer.save_valid_operates)

		# TODO: 否则，下一位玩家摸牌
		else:
			# 更新玩家ID
			self.last_player_id = curr_player.player_id
			self.curr_player_id = (self.last_player_id + 1) % 3
			next_player = players[self.curr_player_id]
			# 摸牌
			self.dealer.mo_cards(next_player)

			valid_actions = self.judge.calc_operates_after_mo_pai(next_player, players, self.dealer, self.last_player_id)

			if valid_actions:
				self.valid_actions = valid_actions
			else:
				self.valid_actions = False

	@staticmethod
	def max_priority_action(save_valid_operates):
		"""
		TODO: 最大优先级
		计算动作优先级
		1.胡开
		2.捡胡
		3.闷胡
		4.暗杠
		5.索杠
		6.明杠
		7.碰
		8.过

		注意: 优先级别最大的动作玩家，则优先选择
		"""
		# tmp_valid_actions临时存储所有玩家有效动作
		result = []
		priority = 0
		for p_id, actions in save_valid_operates.items():
			for action in actions:
				if ACTION_PRIORITY[action] > priority:
					priority = ACTION_PRIORITY[action]
			result.append((p_id, priority))
			priority = 0

		# 优先级别最大的先进行操作
		result = sorted(result, key=lambda v: v[1], reverse=True)

		return result[0][0], save_valid_operates[result[0][0]]


	def get_state(self, players, player_id, other_hand_cards, hand_card_nums):
		"""
		获取玩家状态信息
		"""
		state = {
			'table': self.dealer.table,
			'player_id': player_id,
			'curr_hand_cards': players[player_id].curr_hand_cards,
			'piles': {p.player_id: p.piles for p in players},
			'other_hand_cards': other_hand_cards,
			'hand_card_nums': hand_card_nums,
			'played_cards': self.dealer.played_cards,
			'action_seq_history': self.dealer.record_action_seq_history,
			**self.calc_actions(players[player_id])
		}
		return state

	def calc_actions(self, curr_player):
		"""
		设置当前动作
		"""
		state = {}
		# 碰、杠(明、索、暗)、胡
		if self.valid_actions:
			state["valid_actions"] = ["operator"]
			state["actions"] = self.valid_actions
			return state
		# 正常出牌(锁🔒牌和不锁🔒牌)
		state["valid_actions"] = ["play"]
		# 处于锁🔒牌状态
		if curr_player.lock_cards:
			state["actions"] = curr_player.out_of_lock_cards()
			return state
		# 未处于锁🔒牌状态
		state["actions"] = curr_player.curr_hand_cards

		return state

	def fan_bird(self):
		"""
		闷、捡、胡翻鸡
		"""
		bird_list = []
		for i in range(self.bird_count):
			if self.dealer.left_count > 0:
				bird_list.append(self.dealer.deck.pop().card_value)
				self.dealer.left_count -= 1
			else:
				break
		return bird_list

	def do_fan_bird_xl_hz(self):
		"""
		血流红中翻鸡
		"""
		fan_cards = self.fan_bird()
		if not fan_cards:
			return
		card = fan_cards[0]
		if card == const.LAI_ZI:
			self.base_ji_bei_lv *= 2
			return

		bird_list = []

		# 默认上鸡
		if card % 10 == 9:
			bird_card = card - 8
		else:
			bird_card = card + 1
		bird_list.append(bird_card)

		# 上下鸡
		if self.shang_xia_ji:
			if card % 10 == 1:
				bird_card = card + 8
			else:
				bird_card = card - 1

		# 添加翻鸡牌
		bird_list.append(bird_card)

		for bird in bird_list:
			self.__record_fan_ji_pai[bird] = self.__record_fan_ji_pai.get(bird, 0) + 1

	def do_fan_bird_fk(self):
		"""
		房卡翻鸡
		根据翻的鸡计算当前鸡牌
		此处计算之后，不必再在玩家对象里面都计算一次
		"""
		fan_ji_set = set()
		if not self.__fk_fan_ji_pai:
			return fan_ji_set, -1
		bird_list = self.fan_bird()
		for ji in bird_list:
			# 本鸡
			if self.__fk_ben_ji:
				fan_ji_set.add(ji)

			# 上下鸡
			if ji % 10 == 9:
				fan_ji_set.add(ji - 8)
			else:
				fan_ji_set.add(ji + 1)

			# 下鸡
			if self.__fk_yao_bai_ji:
				if ji % 10 == 1:
					fan_ji_set.add(ji + 8)
				else:
					fan_ji_set.add(ji - 1)

		return fan_ji_set, bird_list[0] if bird_list else - 1

	def deal_ze_ren_ji(self, players):
		"""
		处理责任鸡(责任鸡: 幺鸡， 责任鸡: 乌骨鸡)
		"""
		# 责任鸡
		if self.dealer.curr_mo_card == const.LAI_ZI and self.__round_first_ji == 0:
			self.__ze_ren_ji_seat_id = self.last_player_id
			self.__ze_ren_ji_win_seat_id = self.curr_player_id
			if self.__cfj_seat_id > 0:
				cf_ji_player = self.calc_curr_player(players, self.__cfj_seat_id)
				cf_ji_player.cf_ji = 0
				self.__cfj_seat_id = 0

			curr_player = self.calc_curr_player(players, self.curr_player_id)
			curr_player.ze_ren_ji = 1
			self.__round_first_ji = 1

			return 1

		# 责任乌骨鸡
		if self.dealer.curr_mo_card == CardType.WU_GU_JI and self.__round_first_ji == 0:
			self.__ze_ren_wgj_seat_id = self.last_player_id
			self.__ze_ren_wgj_win_seat_id = self.curr_player_id
			if self.__cf_wgj_seat_id > 0:
				cf_wgj_player = self.calc_curr_player(players, self.__cf_wgj_seat_id)
				cf_wgj_player.cf_wg_ji = 0
				self.__cf_wgj_seat_id = 0

			curr_player = self.calc_curr_player(players, self.curr_player_id)
			curr_player.ze_ren_wgj = 1
			self.__round_first_wgj = 1
			return 2

		return 0

	def check_cfj_xl_hz(self, players, result):
		"""
		检查冲锋鸡
		1.冲锋鸡(幺鸡)
		2.冲锋鸡(乌骨鸡)
		3.冲锋鸡(金鸡 -> 幺鸡)
		"""
		# 冲锋鸡(幺鸡、乌骨鸡)
		if self.__cfj_seat_id:
			ji_card = CardType.YAO_JI
			key = JiType_XL.CF_JI
			self.concreteness_check_cfj(players, result, self.__cfj_seat_id, ji_card, key)
		if self.__cf_wgj_seat_id:
			ji_card = CardType.WU_GU_JI
			key = JiType_XL.WU_GU_CFJ
			self.concreteness_check_cfj(players, result, self.__cf_wgj_seat_id, ji_card, key)

	def concreteness_check_cfj(self, players, result, cfj_seat_id, ji_card, key):
		"""
		具体处理冲锋鸡的分
		若冲锋鸡玩家未叫牌，则需要包鸡给其他所有其余叫牌玩家
		此处单独处理(冲锋鸡/包冲锋鸡)
		"""
		# 当前操作玩家
		curr_player = self.calc_curr_player(players, cfj_seat_id)
		if curr_player.player_id in curr_player.sj_gang_seats:
			return

		score = const.JI_PAI_SCORES_XL.get(key)
		ji_count = self.__record_fan_ji_pai.get(ji_card, 0)
		if ji_count > 0:
			# 冲锋鸡 = (基础幺鸡分 + 翻牌鸡指向幺鸡数量) * 冲锋鸡基础分
			score = (const.JI_PAI_SCORES_XL.get(ji_card) + ji_count) * score
		score *= self.base_ji_bei_lv

		if curr_player.call_pai <= 0:
			for other_player in players:
				if not other_player or other_player.is_out:
					continue
				if other_player.call_pai <= 0:
					continue
				if other_player == curr_player:
					continue
				# 更新玩家分数
				self.update_result_score(result, other_player.player_id, key, score)
				self.update_result_score(result, curr_player.player_id, key, -score)
		else:
			for other_player in players:
				if not other_player or other_player.is_out:
					continue
				if other_player == curr_player:
					continue
				# 更新玩家分数
				self.update_result_score(result, other_player.player_id, key, -score)
				self.update_result_score(result, curr_player.player_id, key, score)

	@staticmethod
	def update_result_score(res, player_id, check_key, scores):
		"""
		记录玩家结算明细(采用积分机制)
		"""
		if scores == 0:
			return
		if not res.get(player_id):
			res[player_id] = {"score": 0, "score_ming_xi": {}}
		res[player_id]["score"] += scores
		res[player_id]["score_ming_xi"][check_key] = res[player_id]["score_ming_xi"].get(check_key, 0) + scores

	def check_zrj_xl_hz(self, players, result):
		"""
		检查责责任鸡接口
		"""
		if self.__ze_ren_ji_seat_id and self.__cfj_seat_id == 0:
			card = CardType.YAO_JI
			key = JiType_XL.ZE_REN_JI
			self.check_out_zrj_xl_hz(players, result, self.__ze_ren_ji_win_seat_id, self.__ze_ren_ji_seat_id, key, card)

		if self.__ze_ren_wgj_seat_id and self.__cf_wgj_seat_id == 0:
			card = CardType.WU_GU_JI
			key = JiType_XL.WU_GU_CFJ
			self.check_out_zrj_xl_hz(players, result, self.__ze_ren_wgj_win_seat_id, self.__ze_ren_wgj_seat_id, key, card)

	def check_out_zrj_xl_hz(self, players, result, zr_win, zr, key, zrj):
		"""
		处理责任鸡计算，由于存在两种责任鸡(幺鸡/乌骨鸡)
		1.第一张打出的幺鸡被碰杠，为责任鸡，该鸡牌结算时赔付3分，其余鸡牌正常结算
		2.责任鸡对其他不是打出冲锋鸡的玩家按一般鸡牌结算

		:param result: 玩家结算明细
		:param zr_win: 则到责任鸡玩家ID
		:param zr: 责任鸡赔付者
		:param key: 计算key(责任鸡)
		:parma zrj: 鸡牌，幺鸡/乌骨鸡
		"""
		# 责任鸡玩家
		zrj_player = self.calc_curr_player(players, zr_win)
		# 责任鸡玩家在烧鸡杠玩家ID中，则直接返回，不进行计算
		if zrj_player in zrj_player.sj_gang_seats:
			return

		# 失去责任鸡玩家
		lose_zrj_player = self.calc_curr_player(players, zr)
		ji_count = 0

		# 暗杠不存在责任鸡，责任鸡为碰/明杠后/索杠
		for pile in zrj_player.piles:
			if zrj not in pile[1]:
				continue
			act = pile[0]
			if act == ActionType.PONG:
				ji_count = 1
				break
			elif act == ActionType.MING_GONG:
				ji_count = 1
				break

			# 可能存在的玩家碰了幺鸡，又索杠幺鸡
			elif act == ActionType.SUO_GONG:
				ji_count = 1
				break

		# 无鸡
		if ji_count == 0:
			return

		# 基础鸡牌分
		base_score = const.JI_PAI_SCORES_XL.get(zrj)
		# 责任鸡分
		score = const.JI_PAI_SCORES_XL.get(key)

		# 若存在翻牌，则为金鸡
		# 翻牌指示数量
		fan_ji_count = self.__record_fan_ji_pai.get(zrj, 0)
		if fan_ji_count > 0:
			base_score += fan_ji_count  # 翻鸡指示胡数量累加(先算好)

		if zrj_player.call_pai > 0:
			# TODO: 先将责任鸡赔付，责任人后续算鸡-1
			# 公式: 责任鸡分数 = (基础幺鸡分数 + 翻牌鸡指向幺鸡数量) * 3 (责任鸡分)
			z_score = score * base_score * self.base_ji_bei_lv
			self.update_result_score(result, zrj_player.player_id, key, z_score)
			self.update_result_score(result, lose_zrj_player.player_id, key, -z_score)

		else:
			if lose_zrj_player.call_pai > 0:
				z_score = score * base_score * self.base_ji_bei_lv
				self.update_result_score(result, zrj_player.player_id, key, z_score)
				self.update_result_score(result, lose_zrj_player.player_id, key, -z_score)

	def do_check_ji_score(self, players, dealer, result):
		"""
		结算鸡分
		1.先算鸡分，为叫牌的玩家包鸡给叫牌玩家
		2.再算冲锋鸡，此处只结算"冲锋鸡"
		3.然后再算责任鸡，责任鸡把得到的责任鸡玩家的所有(包括手牌，打出，碰杠，闷)、幺鸡、乌骨鸡结算，后续不再结算
		4.最后结算翻鸡牌，叫牌玩家与其余玩家(未叫牌/叫牌)，分别结算
		注意: 此处叫牌玩家和胡过牌的玩家统称未叫牌玩家
		"""
		for player in players:
			if not player or player.is_out:
				continue
			if player.player_id in player.sj_gang_seats:
				continue
			if player.call_pai <= 0:
				# 手中打出或打出幺鸡/乌骨鸡
				bao_ji = self.get_all_bird_list(dealer, player, const.DEFAULT_JI_CARDS)
				# 不存在包鸡，则continue
				if not bao_ji:
					continue
				if len(bao_ji) > 2:
					continue
				for other_player in players:
					if not other_player or other_player.is_out:
						continue
					if other_player.player_id == player.player_id:
						continue
					# 判断已包含当前玩家
					if other_player.call_pai <= 0:
						continue
					for j, count in bao_ji.items():
						if j == CardType.YAO_JI:
							key = CardType.YAO_JI
							# 冲锋鸡不在此处包
							if player.cf_ji == 1:
								count -= 1
							# 责任鸡不在此处包
							elif player.ze_ren_ji == 1 and other_player.player_id == self.__ze_ren_ji_seat_id:
								count -= 1
						else:
							key = CardType.WU_GU_JI
							# 冲锋鸡不在此处包
							if player.cf_ji == 1:
								count -= 1
							# 责任鸡不在此处赔付
							elif player.wg_zr_ji == 1 and other_player.player_id == self.__ze_ren_wgj_seat_id:
								count -= 1

						# 基础分数
						base_score = const.JI_PAI_SCORES_XL.get(j)
						fan_count = self.__record_fan_ji_pai.get(j, 0)

						# 翻鸡次数
						if fan_count:
							base_score += fan_count

						score = count * base_score * self.base_ji_bei_lv
						self.update_result_score(result, other_player.player_id, key, score)
						self.update_result_score(result, player.player_id, key, -score)

		# 冲锋鸡(幺鸡、乌骨鸡)
		self.check_cfj_xl_hz(players, result)

		# 责任鸡(幺鸡、乌骨鸡)
		self.check_zrj_xl_hz(players, result)

		# 算翻的鸡(金鸡、银鸡)，此时算叫牌玩家的翻鸡分，未叫牌玩家包鸡已结算过
		# 责任鸡已结算，此处不结算
		record_ji_pai = self.__record_fan_ji_pai

		# 默认鸡
		default_birds = const.DEFAULT_JI_CARDS.copy()
		check_key = JiType_XL.FAN_PAI_JI  # 大的结算Key

		# 多对多，叫牌玩家对其余所有玩家，叫牌玩家间各赔各的
		for player in players:
			if not player or player.is_out:
				continue
			if player.call_pai <= 0:
				continue
			if player.player_id in player.sj_gang_seats:
				continue
			# 计算所有玩家翻鸡牌(手中的|打出的|闷捡胡的|碰杠的)
			all_birds = self.get_all_fan_ji(dealer, player, record_ji_pai, default_birds)
			if not all_birds:
				continue
			for other_player in players:
				if not other_player or other_player.is_out:
					continue
				if player.player_id == other_player.player_id:
					continue
				for j, n in all_birds.items():
					# 鸡牌被翻次数
					fan_count = record_ji_pai.get(j, 0)
					calc_ji_count = n  # 手中鸡的个数
					if j == CardType.YAO_JI:
						# 如果有冲锋鸡，则冲锋鸡已经结算过，此处减去1个冲锋鸡
						if player.cf_ji:
							calc_ji_count -= 1
						# 责任鸡已把所有得到责任鸡所有幺鸡结算
						elif player.ze_ren_ji == 1 and other_player.player_id == self.__ze_ren_ji_seat_id:
							calc_ji_count -= 1
						key = CardType.YAO_JI
					elif j == CardType.WU_GU_JI:
						# 如果有冲锋鸡，则冲锋鸡已经结算过，此处减去1个冲锋鸡
						if player.cf_ji:
							calc_ji_count -= 1
						elif player.wg_zr_ji and other_player.player_id == self.__ze_ren_wgj_seat_id:
							calc_ji_count -= 1
						key = CardType.WU_GU_JI
					else:
						key = JiType_XL.FAN_PAI_JI

					if calc_ji_count <= 0:
						continue
					score = const.JI_PAI_SCORES_XL.get(key, 0)  # 鸡牌基础分
					if fan_count > 1:
						score = score + fan_count  # 改变基础分
						# 翻牌的鸡本省没有分数，翻到才算鸡分，所以-1
						if key == JiType_XL.FAN_PAI_JI:
							score -= 1

					score *= calc_ji_count  # 基础分 * 自己的鸡牌数量
					# 翻到红中，不产生新的鸡牌，且幺鸡和乌骨鸡分数翻倍(当前作废)
					if self.base_ji_bei_lv > 1 and j in default_birds:
						score *= self.base_ji_bei_lv

					# 更新结算明细
					self.update_result_score(result, other_player.player_id, check_key, -score)
					self.update_result_score(result, player.player_id, check_key, score)

	@staticmethod
	def get_all_bird_list(dealer, curr_player, bird_dict, need_men=False):
		"""
		计算包鸡数量
		获取玩家手中以及打出去的 + 碰杠幺鸡或乌骨鸡
		"""
		bao_dict = {}

		# 手中 + 打出去的
		all_cards = dealer.table + curr_player.curr_hand_cards
		for card in all_cards:
			if card in bird_dict:
				bao_dict[card] = bao_dict.get(card, 0) + 1

		# 碰杠
		for combo in curr_player.piles:
			if combo[0] == ActionType.PONG:
				count = 3
			else:
				count = 4
			card = combo[1][-1]
			if card in bird_dict:
				bao_dict[card] = bao_dict.get(card, 0) + count

		if need_men:
			for men_data in curr_player.men_card_infos:
				card = men_data["card"]
				if card in bird_dict:
					bao_dict[card] = bao_dict.get(card, 0) + 1

		# print("输出鸡牌: ", bao_dict)

		return bao_dict

	@staticmethod
	def get_all_fan_ji(dealer, curr_player, judge_fan_birds: dict, default_birds: list):
		"""
		获取游戏中所有的鸡牌: 碰、杠、闷
		judge_fan_birds: 桌子中的翻鸡
		"""
		fan_ji_dict = dict()
		for card in curr_player.curr_hand_cards:
			if judge_fan_birds.get(card):
				fan_ji_dict[card] = fan_ji_dict.get(card, 0) + 1

		for card in dealer.table:
			# 打出的只算默认鸡
			if card in default_birds:
				fan_ji_dict[card] = fan_ji_dict.get(card, 0) + 1

		for men_data in curr_player.men_card_infos:
			card = men_data["men_card"]
			if card in default_birds:
				fan_ji_dict[card] = fan_ji_dict.get(card, 0) + 1
			elif judge_fan_birds.get(card):
				fan_ji_dict[card] = fan_ji_dict.get(card, 0) + 1

		# 碰、杠
		for combo in curr_player.piles:
			if combo[0] == ActionType.PONG:
				count = 3
			else:
				count = 4
			card = combo[1][0]
			if card in default_birds:
				fan_ji_dict[card] = fan_ji_dict.get(card, 0) + count
			elif judge_fan_birds.get(card):
				fan_ji_dict[card] = fan_ji_dict.get(card, 0) + count

		return fan_ji_dict

	def do_check_by_xl_hz(self, players, dealer, result: dict):
		"""
		主要结算该局中鸡牌分数
		"""
		self.do_check_ji_score(players, dealer, result)

	def do_check_by_liu_ju_xl_hz(self, players, dealer, result):
		"""
		流局计算鸡牌分数
		1.查叫分
		2.闷捡分
		"""
		# 1.查叫分(p为叫牌玩家，other_p为未叫牌玩家)
		for player in players:
			if not player or player.is_out:
				continue
			if player.call_pai <= 0:
				continue
			key = player.call_pai
			for other_player in players:
				if not other_player or other_player.is_out:
					continue
				if other_player.call_pai > 0:
					continue
				score = const.HU_PAI_SCORES.get(key) or 0
				self.update_result_score(result, other_player.player_id, key, -score)
				self.update_result_score(result, player.player_id, key, score)
		# 结算鸡分
		self.do_check_ji_score(players, dealer, result)

	def check_cfj_fk(self, players, result, fan_ji_cards=None, liu_ju=False):
		"""
		结算冲峰鸡
		"""
		if not self.__fk_cfj:
			return
		if not fan_ji_cards:
			fan_ji_cards = []
		if self.__cfj_seat_id != 0:
			ji_card = CardType.YAO_JI
			key = JiType_FK.CF_JI
			if ji_card in fan_ji_cards:  # 翻到为金鸡
				key = JiType_FK.CF_JIN_JI
			score = const.JI_PAI_SCORE_FK.get(key)
			self.concreteness_check_cfj_fk(players, result, score, ji_card, self.__cfj_seat_id, liu_ju)

		if self.__cf_wgj_seat_id != 0:
			ji_card = CardType.WU_GU_JI
			key = JiType_FK.WU_GU_CFJ
			if ji_card in fan_ji_cards:  # 翻到金鸡
				key = JiType_FK.WU_GU_CF_JIN_JI  # 乌骨冲峰捡鸡
			score = const.JI_PAI_SCORE_FK.get(key, 0)
			self.concreteness_check_cfj_fk(players, result, score, ji_card, self.__cf_wgj_seat_id, liu_ju)

	def concreteness_check_cfj_fk(self, players, result, score, card, cfj_seat_id, liu_ju):
		"""
		具体处理冲锋鸡的分
		此处只处理鸡牌
		"""
		cfj_player = self.calc_curr_player(players, cfj_seat_id)
		if cfj_player in cfj_player.sj_gang_seats:
			return
		# 处理鸡牌分
		if cfj_player.call_pai > 0:
			win_from = []
			total_score = 0
			for other_player in players:
				if not other_player:
					continue
				if other_player.player_id == cfj_player.player_id:
					continue
				win_from.append(other_player.player_id)
				other_data = other_ming_xi_data(cfj_seat_id, -score, card)
				update_result_score(result, other_player.player_id, 0, other_data)
				total_score += score
			self_data = self_ming_xi_data(win_from, total_score, card)
			update_result_score(result, cfj_seat_id, 1, self_data)
		else:
			win_from = []
			total_score = 0
			for other_player in players:
				if not other_player:
					continue
				if other_player.player_id == cfj_seat_id:
					continue
				if other_player.call_pai <= 0:
					continue
				# 包给叫牌玩家
				win_from.append(other_player.player_id)
				other_data = other_ming_xi_data(cfj_seat_id, score, card)
				update_result_score(result, other_player.player_id, 0, other_data)
				total_score += score
			self_data = self_ming_xi_data(win_from, -total_score, card)
			update_result_score(result, cfj_seat_id, 1, self_data)

	def check_zrj_fk(self, players, result, liu_ju=False):
		"""
		结算责任鸡
		责任鸡无金鸡一说，额外算分
		"""
		if not self.__fk_ze_ren_ji:
			return
		if self.__ze_ren_ji_seat_id != 0 and self.__ze_ren_ji_win_seat_id != 0:
			ji_card = CardType.YAO_JI
			key = JiType_FK.ZE_REN_JI
			score = const.JI_PAI_SCORE_FK.get(key, 0)
			self.concreteness_check_zrj_fk(players, result, self.__ze_ren_ji_win_seat_id, self.__ze_ren_ji_seat_id, ji_card, score, liu_ju)

		if self.__ze_ren_wgj_seat_id != 0 and self.__ze_ren_wgj_win_seat_id != 0:
			ji_card = CardType.WU_GU_JI
			key = JiType_FK.WU_GU_ZRJ
			score = const.JI_PAI_SCORE_FK.get(key, 0)
			self.concreteness_check_zrj_fk(players, result, self.__ze_ren_ji_win_seat_id, self.__ze_ren_ji_seat_id, ji_card, score, liu_ju)

	def concreteness_check_zrj_fk(self, players, result, ze_ren_win, ze_ren_lose, card, score, liu_ju):
		"""
		具体处理责任鸡算分
		"""
		get_zrj_player = self.calc_curr_player(players, ze_ren_win)
		lose_zrj_player = self.calc_curr_player(players, ze_ren_lose)
		if get_zrj_player.player_id in get_zrj_player.sj_gang_seats:
			return
		if get_zrj_player.call_pai > 0:
			if liu_ju:
				return
			other_data = other_ming_xi_data(ze_ren_win, -score, card)
			self_data = self.action_ming_gang([ze_ren_lose], score, card)
			update_result_score(result, ze_ren_lose, 0, other_data)
			update_result_score(result, ze_ren_win, 1, self_data)
		else:
			# 责任鸡玩家需要叫牌且未诈胡(此处无诈胡)
			if lose_zrj_player.call_pai > 0:
				other_data = other_ming_xi_data(ze_ren_win, -score, card)
				self_data = self_ming_xi_data([ze_ren_lose], score, card)
				update_result_score(result, ze_ren_lose, 0, other_data)
				update_result_score(result, ze_ren_win, 1, self_data)

	def bao_ji_check_fk(self, players, result, liu_ju=False):
		"""
		结算包鸡
		流局未听牌需包鸡
		"""
		for player in players:
			if not player:
				continue
			if player.player_id in player.sj_gang_seats:
				continue
			if player.call_pai > 0:
				continue
			# 计算所有鸡牌
			player.calc_all_ji_cards(self.__fk_default_ji, self.dealer,  with_out=self.__fk_man_tang_ji)
			p_ji_cards = player.fk_ji_cards[:]
			# 计算站鸡
			p_stand_ji = player.calc_stand_ji(self.__fk_default_ji)

			# 冲锋鸡之前算过
			if self.__cfj_seat_id != 0 and self.__cfj_seat_id == player.player_id:
				if CardType.YAO_JI in p_ji_cards:
					p_ji_cards.remove(CardType.YAO_JI)
			if self.__cf_wgj_seat_id != 0 and self.__cf_wgj_seat_id == player.player_id:
				if CardType.WU_GU_JI in p_ji_cards:
					p_ji_cards.remove(CardType.WU_GU_JI)
			if not p_ji_cards:
				continue

			ji_card_count = card_count_fk(p_ji_cards)
			stand_ji_card_count = card_count_fk(p_stand_ji)

			for ji, count in ji_card_count.items():
				bei_lv = 1
				score = 0
				# 金鸡 x 2, 站鸡 x 2
				if self.__fk_zj:
					stand_ji_count = stand_ji_card_count.get(ji, 0)
					if stand_ji_count > 0:
						count -= stand_ji_count
						# 第一个x2是承担2份，第二个x2是站鸡翻倍
						score = const.JI_PAI_SCORE_FK.get(ji, 1) * stand_ji_count * 2 * bei_lv * 2

				win_total = 0
				win_from = []
				for other_player in players:
					if not other_player:
						continue
					if other_player.player_id == player.player_id:
						continue
					if other_player.call_pai <= 0:
						continue
					if other_player.player_id in other_player.sj_gang_seats:
						continue
					ori_score = score
					ori_score += const.JI_PAI_SCORE_FK.get(ji, 1) * count * bei_lv
					other_data = other_ming_xi_data(player.player_id, score, ji)
					update_result_score(result, other_player.player_id, 0, other_data)
					win_total += ori_score
					win_from.append(other_player.player_id)

				self_data = self_ming_xi_data(win_from, -win_total, ji)
				update_result_score(result, player.player_id, 1, self_data)

	@staticmethod
	def compare_cards_type_is_get_qing_yi_se(hu_type, is_zi_mo=False):
		"""
		牌型分数低于清一色算清一色，牌型分数高于清一色算法实际牌型
		此接口只判断玩家牌型是否小于清一色
		"""
		card_type_score = const.HU_PAI_SCORES.get(hu_type, 0)
		qys_score = const.HU_PAI_SCORES.get(const.HuPaiType.QING_YI_SE)
		flag = True
		if card_type_score < qys_score:
			flag = False
			card_type_score = qys_score
		if is_zi_mo:
			card_type_score *= 2

		return flag, card_type_score

	def do_check_over_liu_ju_fk(self, players, result):
		"""
		3人流局结算(房卡场)
		流局查叫算分(牌型分)
		外循环控制叫牌且未诈胡玩家，内循环控制未叫牌玩家
		"""
		call_pai_players = []  # 叫牌玩家
		no_call_pai_player = []  # 没叫牌玩家
		for player in players:
			if not player or player.is_out:
				continue
			# 未叫牌玩家
			if player.call_pai <= 0:
				no_call_pai_player.append(player.player_id)
			else:
				call_pai_players.append(player.player_id)

		if 0 < len(call_pai_players) < len(players):
			flag = False  # 表示是否同时存在未叫牌玩家
			if no_call_pai_player:
				flag = True
			for p_id in call_pai_players:
				player = self.calc_curr_player(players, p_id)
				key = player.call_pai  # 叫牌类型(默认为0)
				score = const.HU_PAI_SCORES[key]
				# 存在叫牌和未叫牌
				if flag:
					# 只有一位玩家未叫牌
					win_score = 0
					if not no_call_pai_player and len(no_call_pai_player) == 1:
						no_call_pai_id = no_call_pai_player[0]
						lost_data = other_ming_xi_data(p_id, score=-score)
						update_result_score(result, no_call_pai_id, 0, lost_data)
						win_score += score

						win_data = self_ming_xi_data([no_call_pai_id], score=win_score)
						update_result_score(result, p_id, 1, win_data)
				else:
					# 两位玩家未叫牌
					for p_id in no_call_pai_player:
						is_gt, score = self.compare_cards_type_is_get_qing_yi_se(key)
						player = self.calc_curr_player(players, p_id)
						lost_data = other_ming_xi_data(p_id, score=-score)
						update_result_score(result, player.player_id, 0, lost_data)

					win_data = self_ming_xi_data(no_call_pai_player, score=score*len(no_call_pai_player))
					update_result_score(result, p_id, 1, win_data)

		# 房卡包鸡
		if self.__fk_bao_ji:
			self.check_cfj_fk(players, result, liu_ju=True)
			self.check_zrj_fk(players, result, liu_ju=True)
			self.bao_ji_check_fk(players, result, liu_ju=True)

		return result

	def do_check_over_kai_hu_fk(self, players, result):
		"""
		房卡，开牌结算
		"""
		fan_ji_cards, zuo_ji = self.do_fan_bird_fk()
		self.__fk_ji_pai = fan_ji_cards

		# 冲锋鸡
		self.check_cfj_fk(players, result, fan_ji_cards)
		# 责任鸡(责任鸡无金鸡一说，额外算分)
		self.check_zrj_fk(players, result)

		# 翻鸡 + 幺鸡(默认鸡) + 乌骨鸡
		fan_bird_list = fan_ji_cards.copy()
		# 先计算正常玩家鸡份(叫牌)
		for player in players:
			if not player:
				continue
			if player.call_pai <= 0:
				continue
			if player.player_id in player.sj_gang_seats:
				continue
			# 计算玩家有几个鸡牌
			player.calc_all_ji_cards(self.__fk_default_ji, self.dealer, fan_bird_list, self.__fk_man_tang_ji)
			p_ji_cards = player.fk_ji_cards[:]
			p_stand_ji = player.calc_stand_ji(self.__fk_default_ji)

			# 冲锋鸡之前算过 -1
			if self.__cfj_seat_id != 0 and self.__cfj_seat_id == player.player_id:
				if CardType.YAO_JI in p_ji_cards:
					p_ji_cards.remove(CardType.YAO_JI)
			if self.__cf_wgj_seat_id != 0 and self.__cf_wgj_seat_id == player.player_id:
				if CardType.WU_GU_JI in p_ji_cards:
					p_ji_cards.remove(CardType.WU_GU_JI)
			if not p_ji_cards:
				continue

			# 鸡牌统计并算分
			ji_card_count = card_count_fk(p_ji_cards)
			stand_ji_card_count = card_count_fk(p_stand_ji)
			for ji, count in ji_card_count.items():
				bei_lv = 1
				score = 0
				if ji in self.__fk_default_ji and ji in fan_bird_list:
					bei_lv = 2
					stand_ji_count = stand_ji_card_count.get(ji, 0)
					if stand_ji_count > 0:
						count -= stand_ji_count
						# x2是站鸡翻倍
						score = const.JI_PAI_SCORE_FK.get(ji, 1) * stand_ji_count * bei_lv * 2
				else:
					win_total = 0
					win_from = []
					for other_player in players:
						if not other_player:
							continue
						if player.player_id == other_player.player_id:
							continue
						ori_score = score
						ori_score += const.JI_PAI_SCORE_FK.get(ji, 1) * count * bei_lv
						other_data = other_ming_xi_data(player.player_id, -score, ji)
						update_result_score(result, other_player.player_id, 0, other_data)
						win_total += ori_score
						win_from.append(other_player.player_id)
					self_data = self_ming_xi_data(win_from, win_total, ji)
					update_result_score(result, player.player_id, 1, self_data)

		return result

	def do_check_over_fk(self, players, over_type):
		"""
		检查房卡结束计算
		"""
		result = {}
		is_liu_ju = over_type == OverType.LIU_JU
		# 流局
		if is_liu_ju:
			return self.do_check_over_liu_ju_fk(players, result)
		# 胡开
		return self.do_check_over_kai_hu_fk(players, result)

	def do_check_over_xl_hz(self, players, dealer, is_hz):
		"""
		血流红中捉鸡结算
		黄庄结算与胡牌结算分开计算
		"""
		scores_result = {}
		# 是否荒庄，默认为True
		if is_hz:
			self.do_check_by_liu_ju_xl_hz(players, dealer, scores_result)
			return scores_result
		# 不是荒庄就是胡开
		else:
			self.do_check_by_xl_hz(players, dealer, scores_result)
			return scores_result

	def deal_first_ji(self, action, curr_player):
		"""
		处理冲锋鸡，其包括(冲锋幺鸡、冲锋乌骨鸡)
		"""
		if action == CardType.YAO_JI and self.__round_first_ji == 0:
			self.__round_first_ji = 1
			self.__cfj_seat_id = curr_player.player_id
			curr_player.cf_ji = 1
		if action == CardType.WU_GU_JI and self.__round_first_wgj == 0:
			self.__round_first_ji = 1
			self.__cf_wgj_seat_id = curr_player.player_id
			curr_player.cf_wg_ji = 1