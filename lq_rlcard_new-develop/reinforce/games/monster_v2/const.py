# -*- coding: utf-8 -*-

import numpy as np

Card2Column = {
	11: 0, 12: 1, 13: 2, 3: 3, 8: 4, 5: 5, 10: 6, 20: 7, 21: 8
}

NumOnes2Array = {
	0: np.array([0, 0, 0, 0]),
	1: np.array([1, 0, 0, 0]),
	2: np.array([1, 1, 0, 0]),
	3: np.array([1, 1, 1, 0]),
	4: np.array([1, 1, 1, 1])
}