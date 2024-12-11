# -*- coding: utf-8 -*-

import subprocess
import sys
# from distutils.version import LooseVersion

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'torch' in installed_packages:
    from rlcards.agents.nfsp_agent import NFSPAgent as NFSPAgent

from rlcards.agents.cfr_agent import CFRAgent
from rlcards.agents.nfsp_agent import NFSPAgent
from rlcards.agents.dqn_agent_en import DQNAgent
from rlcards.agents.random_agent import RandomAgent
