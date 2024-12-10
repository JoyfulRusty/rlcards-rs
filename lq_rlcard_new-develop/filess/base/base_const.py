# -*- coding: utf-8 -*-

import sys
from enum import unique
from base.base_enum import BaseEnum, BaseStrEnum


python_version = sys.version
# python3.10版本使用
if python_version < "3.11":
	StrEnum = BaseStrEnum
else:
	from enum import StrEnum


@unique
class ServiceEnum(BaseEnum):
	"""
	服务枚举号
	"""
	# 机器人 -> 子游戏，子服务游戏枚举[1 - 99]，推送到游戏
	C_MONSTER = 5, "monster", "子服务游戏-打妖怪"
	C_WATER_FISH = 6, "water_fish", "子服务游戏-暗水鱼"
	C_LANDLORDS = 7, "landlords", "子服务游戏-斗地主"

	# 子游戏 -> 机器人，子服务游戏枚举[101 - 199]，接收游戏发送
	ROBOT_MONSTER = 101, "monster", "打妖怪机器人"
	ROBOT_WATER_FISH = 102, "water_fish", "暗水鱼机器人"
	ROBOT_LANDLORDS = 103, "landlords", "斗地主机器人"


print(ServiceEnum.ROBOT_LANDLORDS.val)
print(ServiceEnum.ROBOT_LANDLORDS.desc)
print(ServiceEnum.ROBOT_LANDLORDS.phrase)


class RobotCmdMethods(BaseEnum):
	"""
	todo: 所有游戏共同使用命令[子游戏 -> 机器人]
	规范所有从子游戏发送过来的回调方法命令
	方法命令[201 - 299]
	"""
	CAL_ACTION = 201, "出牌"
	CAL_PONG = 202, "麻将碰"
	CAL_GONG = 203, "麻将杠"
	CAL_SP = 204, "水鱼撒扑"
	CAL_FARMER_LP = 205, "水鱼闲家亮牌"
	CAL_FARMER_QG = 206, "水鱼闲家强攻"
	CAL_FARMER_MI = 207, "水鱼闲家密"
	CAL_LANDLORD_SHA = 208, "水鱼庄家杀"
	CAL_LANDLORD_ZOU = 209, "水鱼庄家走"
	CAL_FARMER_XIN = 210, "水鱼闲家信"
	CAL_FARMER_FAN = 211, "水鱼闲家反"
	CAL_WORLD_CHAT = 212, "世界聊天"
	CAL_PYH_CHAT = 213, "牌友会聊天"
	CAL_SINGLE_CHAT = 214, "单聊"
	CAL_BID = 215, "斗地主叫牌"
	CAL_CHAN = 216, "斗地主铲"
	CAL_DO_CHAN = 217, "斗地主反铲"
	CAL_EXCHANGE_THREE = 218, "斗地主换三张"
	CAL_EXCHANGE_FIVE = 219, "斗地主换五张"


class CmdRoomMethods(BaseEnum):
	"""
	todo: 打妖怪游戏[机器人 -> 子游戏]
	规范预测后，子游戏对应操作命令
	"""
	NEW_MATCH = 1, "新匹配", "该命令号服务内部使用"
	LOST_CONNECT = 2, "掉线", "该命令号服务内部使用, 通知子服务玩家掉线"
	ENTER_ROOM = 3, "进入房间"
	ROOM_INFO = 4, "房间信息"
	PLAYER_INFO = 5, "玩家信息"
	QUIT_ROOM = 6, "离开房间"
	READY = 7, "准备/取消准备"
	TRUSTEE = 8, "托管"
	ROUND_START = 9, "游戏开始"
	DEALER_CARDS = 10, "发牌"
	PLAY_CARDS = 11, "打牌"
	PICK_CARDS = 12, "捡/收牌"
	GIVE_UP = 13, "认输"
	TURN_TO = 14, "轮到"
	TURN_END = 15, "一圈结束"
	GO_BROKE = 16, "破产"
	GOLD_NOT_ENOUGH = 17, "豆子不足"
	RECHARGE = 18, "充值"
	BROADCAST_CHAT = 19, "广播聊天"
	FORCE_DISMISS = 20, "强制解散", "该命令号服务内部使用(应对流程卡住的桌子)"
	ROUND_OVER = 21, "一局结束"
	GAME_OVER = 22, "游戏结束"
	DEDUCT_TICKETS = 23, "扣门票"

	CONFIRM_DEALER = 30, "确定庄家", "水鱼"
	SA_PU = 31, "撒扑", "水鱼"
	START_BET = 32, "开始下注", "水鱼"
	DO_BET = 33, "玩家下注", "水鱼"
	DEALER_OPERATE = 34, "庄操作", "水鱼"
	FARMER_OPERATE = 35, "闲操作", "水鱼"

	START_BID = 40, "开始叫分", "斗地主"
	TURN_BID = 41, "轮到叫分", "斗地主"
	DO_BID = 42, "do 叫分", "斗地主"
	START_REDOUBLE = 43, "开始加倍", "斗地主（铲？）"
	TURN_REDOUBLE = 44, "轮到加倍"
	DO_REDOUBLE = 45, "do 加倍", "斗地主（铲？）"
	START_RE_REDOUBLE = 46, "开始加加倍", "斗地主（反铲）"
	TURN_RE_REDOUBLE = 47, "轮到加加倍", "斗地主（反铲）"
	DO_RE_REDOUBLE = 48, "DO 加加倍", "斗地主（反铲）"

class RmqGameChannel(StrEnum):
	"""
	机器人推送频道
	"""
	C_SERVICES = "C_SERVICES"