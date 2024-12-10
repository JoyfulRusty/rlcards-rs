# -*- coding: utf-8 -*-

import logging

handle = logging.StreamHandler()
handle.setFormatter(logging.Formatter('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'))

logger = logging.getLogger('dmc_v2')
logger.propagate = False
logger.addHandler(handle)
logger.setLevel(logging.INFO)