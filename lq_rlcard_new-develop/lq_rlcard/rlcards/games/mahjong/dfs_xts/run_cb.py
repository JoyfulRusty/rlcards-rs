# -*- coding: utf-8 -*-

import split

from rlcards.utils.utils import cal_time

@cal_time
def main():
	cards = [1, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1]
	res = split.get_hu_info(cards, 24, 27)
	if res:
		print("hu le")
	else:
		print("can't hu")


if __name__ == "__main__":
	main()
