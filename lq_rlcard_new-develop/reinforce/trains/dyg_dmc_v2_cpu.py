# -*- coding: utf-8 -*-

from reinforce.dmc_v2.dmc import DmcTrainer
from reinforce.env.v2.monster import MonsterEnv as Env


def dmc_train():
	"""
	Initialize the dmc training Environment
	"""
	trainer = DmcTrainer(
		x_pid="monster_cpu",
		save_interval=120,
		training_device='cpu',
		load_model=True,
		exp_epsilon=0.1,
		exp_decay=1.0,
		batch_size=32,
		unroll_length=100,
		num_buffers=100,
		num_threads=3,  # alter run threads nums
		max_grad_norm=40.0,
		learn_rate=0.0001,
		alpha=0.99,
		momentum=0,
		epsilon=1e-5,
		positions=["down", "right", "up", "left"],  # positions -> ["landlord_up", "landlord", "landlord_down"]
		action_dim=33,
		no_action_dim=385,
		z_y_dim=12,
		z_x_dim=33
	)

	# Initialize the training environment
	trainer.init_env(Env(env="monster", device="cpu"))

	# Running the start training
	trainer.start()


if __name__ == '__main__':
	dmc_train()