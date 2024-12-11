# -*- coding: utf-8 -*-

import pygame


class Button:
	"""
	创建按钮button
	"""
	def __init__(self, screen, msg, left, top):
		"""
		初始化按钮属性参数
		msg -> 为要在按钮中显示的文本信息
		"""
		self.msg_img = None
		self.msg_img_rect = None
		self.screen = screen
		self.screen_rect = screen.get_rect()
		self.height, self.width = 50, 150
		self.button_color = (71, 61, 139)  # 设置按钮的rect对象颜色为深蓝
		self.text_color = (255, 255, 255)  # 设置文本的颜色为白色

		# 初始化字体
		pygame.font.init()
		self.font = pygame.font.SysFont("kaiti", 20)  # 设置文本为默认字体，字号为20
		self.rect = pygame.Rect(0, 0, self.width, self.height)
		# self.rect.center = self.screen_rect.center  # 创建按钮的rect对象，并使其居中
		self.left = left
		self.top = top
		self.deal_msg(msg)  # 渲染图像

	def deal_msg(self, msg):
		"""
		将msg渲染为图像，并将其在按钮上居中
		"""
		# render将存储在msg的文本转换为图像
		self.msg_img = self.font.render(msg, True, self.text_color, self.button_color)
		# 根据文本图像创建一个rect
		self.msg_img_rect = self.msg_img.get_rect()
		# 将该rect的center属性设置为按钮的center属性
		self.msg_img_rect.center = self.rect.center

	def draw_button(self):
		"""
		绘制按钮
		"""
		# 填充颜色
		# self.screen.fill(self.button_color, self.rect)
		# 将该图像绘制到屏幕
		self.screen.blit(self.msg_img, (self.left, self.top))

	def is_click(self):
		"""
		判断是否点击
		"""
		point_x, point_y = pygame.mouse.get_pos()
		x = self.left
		y = self.top
		w, h = self.msg_img.get_size()
		in_x = x < point_x < x + w
		in_y = y < point_y < y + h
		return in_x and in_y