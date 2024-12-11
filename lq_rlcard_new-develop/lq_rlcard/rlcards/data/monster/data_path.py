# -*- coding: utf-8 -*-

import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

CARD_TYPE_PATH = os.path.join(BASE_PATH, 'monster', 'jsondata', 'card_type_singal.json')
ACTION_SPACE_PATH = os.path.join(BASE_PATH, 'monster', 'jsondata', 'action_space_singal.txt')
TYPE_CARD_PATH = os.path.join(BASE_PATH, 'monster', 'jsondata', 'type_card_singal.json')


if __name__ == '__main__':
	print('card_type: ', CARD_TYPE_PATH)
	print('action_space: ', ACTION_SPACE_PATH)
	print('type_card: ', TYPE_CARD_PATH)