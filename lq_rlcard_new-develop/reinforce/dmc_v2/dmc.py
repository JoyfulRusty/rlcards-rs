# -*- coding: utf-8 -*-

import os
import pprint
import threading
import time
import torch
import timeit
import typing
import traceback

from torch import nn
from collections import deque
from torch import multiprocessing as mp

from reinforce.radam import RAdam
from reinforce.ranger.ranger20 import Ranger
from reinforce.dmc_v2.logger import logger as Logger
from reinforce.dmc_v2.file_writer import FileWriter
from reinforce.dmc_v2.model.model_v2_res import Model

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


class DmcTrainer:
	"""
	Deep Monte Carlo trainer.
	"""

	def __init__(
			self,
			x_pid="monster",
			save_interval=120,
			training_device='cpu',
			load_model=True,
			save_dir="results/dmc_v2",
			total_frames=100000000000,
			exp_epsilon=0.1,
			exp_decay=1.0,
			batch_size=32,
			unroll_length=100,
			num_buffers=50,
			num_threads=1,
			max_grad_norm=40.0,
			learn_rate=0.0001,
			alpha=0.99,
			momentum=0.45,
			epsilon=1e-8,
			positions=None,  # positions -> ["landlord_up", "landlord", "landlord_down"]
			action_dim=30,
			no_action_dim=351,
			z_y_dim=4,
			z_x_dim=30
		):
		"""
		Initializes the DMC trainer.
		"""
		self.env = None
		self.x_pid = x_pid
		self.save_interval = save_interval
		self.training_device = training_device
		self.load_model = load_model
		self.save_dir = save_dir
		self.total_frames = total_frames
		self.exp_epsilon = exp_epsilon
		self.exp_decay = exp_decay
		self.batch_size = batch_size
		self.unroll_length = unroll_length
		self.num_buffers = num_buffers
		self.num_threads = num_threads
		self.max_grad_norm = max_grad_norm
		self.learn_rate = learn_rate
		self.alpha = alpha
		self.momentum = momentum
		self.epsilon = epsilon
		self.positions = positions
		self.action_dim = action_dim
		self.no_action_dim = no_action_dim
		self.z_y_dim = z_y_dim
		self.z_x_dim = z_x_dim
		self.checkpoint_path = os.path.expandvars(os.path.expanduser('%s/%s/%s' % (save_dir, x_pid, 'model.tar')))
		self.file_log = FileWriter(x_pid=self.x_pid, root_dir=self.save_dir)
		self.mean_episode_return_buf = {position: deque(maxlen=100) for position in self.positions}

		# set devices: cuda:0 or cpu
		self.device_iterator = ["cpu"]
		self.training_device = "cpu"

		# if torch.cuda.is_available() and self.training_device != "cpu":
		if torch.cuda.is_available() and self.training_device != "cpu":
			self.device_iterator = ["0"]
			self.training_device = "0"

	def init_env(self, env):
		"""
		Initialize the training environment
		"""
		self.env = env

	@staticmethod
	def compute_loss(output_vals, target_vals):
		"""
		Compute loss for training and evaluation
		"""
		loss = ((output_vals.squeeze(-1) - target_vals) ** 2).mean()
		return loss

	def learn(self, position, actor_models, learn_model, batch, optimizer, position_lock):
		"""
		Learn the training and evaluation
		"""
		# Judge Cuda is available or not available
		device = torch.device('cuda:' + str(self.training_device) if self.training_device != "cpu" else "cpu")
		if torch.cuda.is_available() and self.training_device != "cpu":
			device = torch.device('cuda:' + str(self.training_device))
		obs_x = torch.cat((batch['obs_x_no_action'].to(device), batch['obs_action'].to(device)), dim=2).float()
		learn_x_batch = torch.flatten(obs_x, 0, 1).float()
		learn_z_batch = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
		target = torch.flatten(batch['target'].to(device), 0, 1).float()
		episode_returns = batch['episode_return'][batch['done']]
		self.mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
		with position_lock:
			output_vals = learn_model.forward(z=learn_z_batch, x=learn_x_batch)
			loss = self.compute_loss(output_vals, target)
			stats = {
				'mean_episode_return_' + position:
					torch.mean(torch.stack([_r for _r in self.mean_episode_return_buf[position]])).item(),
				'loss_' + position: loss.item(),
			}
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(learn_model.parameters(), self.max_grad_norm)
			optimizer.step()
			for actor_model in actor_models.values():
				actor_model.get_model(position).load_state_dict(learn_model.state_dict())
			return stats

	def create_buffer(self):
		"""
		Create buffer for training and evaluation
		"""
		buffers = {}
		for device in self.device_iterator:
			buffers[device] = {}
			for position in self.positions:
				specs = dict(
					done=dict(size=(self.unroll_length, ), dtype=torch.bool),
					episode_return=dict(size=(self.unroll_length, ), dtype=torch.float32),
					target=dict(size=(self.unroll_length, ), dtype=torch.float32),
					obs_x_no_action=dict(size=(self.unroll_length, self.no_action_dim), dtype=torch.int8),
					obs_action=dict(size=(self.unroll_length, self.action_dim), dtype=torch.int8),
					obs_z=dict(size=(self.unroll_length, self.z_y_dim, self.z_x_dim), dtype=torch.int8),
				)
				_buffers: Buffers = {key: [] for key in specs}
				for _ in range(self.num_buffers):
					for key in _buffers:
						if not device == "cpu":
							_buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
						else:
							_buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
						_buffers[key].append(_buffer)
					buffers[device][position] = _buffers
		return buffers

	def create_optimizers(self, learner_model):
		"""
		Create optimizers
		:param learner_model: learn model
		"""
		optimizers = {}
		for position in self.positions:
			optimizer = Ranger(learner_model.parameters(position), lr=self.learn_rate)
			optimizers[position] = optimizer
		return optimizers

	def create_optimizers_by_rad(self, learner_model):
		"""
		Create optimizers by LookAhead
		"""
		optimizers = {}
		for position in self.positions:
			init_optimizer = RAdam(
				learner_model.parameters(position),
				lr=self.learn_rate,
				eps=self.epsilon)
			optimizers[position] = init_optimizer
		return optimizers

	def create_optimizers_by_rms(self, learner_model):
		"""
		Create optimizers by RMS
		"""
		optimizers = {}
		for position in self.positions:
			init_optimizer = torch.optim.RMSprop(
				learner_model.parameters(position),
				lr=self.learn_rate,
				momentum=self.momentum,
				eps=self.epsilon,
				alpha=self.alpha)
			optimizers[position] = init_optimizer
		return optimizers

	def create_lr_scheduler(self, optimizers):
		"""
		Create learning rate scheduler
		:param optimizers: optimizers
		"""
		schedulers = {}
		for position in self.positions:
			scheduler = torch.optim.lr_scheduler.StepLR(
				optimizers[position],
				step_size=self.batch_size * self.unroll_length,
				gamma=0.882,
				last_epoch=-1
			)
			schedulers[position] = scheduler
		return schedulers

	def get_batch(self, free_queue, full_queue, buffers, lock):
		"""
		Get batch of training parameters
		:param free_queue: free queue
		:param full_queue: full queue
		:param buffers: buffers
		:param lock: lock
		"""
		with lock:
			indices = [full_queue.get() for _ in range(self.batch_size)]
		batch = {key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers}
		for m in indices:
			free_queue.put(m)
		return batch

	def start(self):
		"""
		Run the distributed training.
		"""
		# Initialize actor models
		models = {}
		for device in self.device_iterator:
			model = Model(device=device, positions=self.positions)
			model.share_memory()
			model.eval()
			models[device] = model

		# Initialize buffers
		buffers = self.create_buffer()

		# Initialize queues
		actor_processes = []
		ctx = mp.get_context('spawn')
		free_queue, full_queue = {}, {}
		for device in self.device_iterator:
			_free_queue = {position: ctx.SimpleQueue() for position in self.positions}
			_full_queue = {position: ctx.SimpleQueue() for position in self.positions}
			free_queue[device] = _free_queue
			full_queue[device] = _full_queue
		# Learner models for training
		learner_model = Model(device=self.training_device, positions=self.positions)
		# Initialize the model with the optimizers and lr schedulers
		optimizers = self.create_optimizers(learner_model)
		# schedulers = self.create_lr_scheduler(optimizers)
		# stat keys
		stat_keys = []
		for position in self.positions:
			stat_keys.append('mean_episode_return_' + str(position))
			stat_keys.append('loss_' + str(position))
		frames, stats = 0, {k: 0 for k in stat_keys}
		# load model checkpoints
		if self.load_model and os.path.exists(self.checkpoint_path):
			checkpoint_states = torch.load(
				self.checkpoint_path, map_location='cuda:' + str(self.training_device) if self.training_device != "cpu" else "cpu")
			for idx, position in enumerate(self.positions):
				learner_model.get_model(position).load_state_dict(checkpoint_states["model_state_dict"][idx])
				optimizers[position].load_state_dict(checkpoint_states["optimizer_state_dict"][idx])
				for device in self.device_iterator:
					models[device].get_model(position).load_state_dict(learner_model.get_model(position).state_dict())
			stats = checkpoint_states["stats"]
			frames = checkpoint_states["frames"]
			Logger.info(f"Resuming preempted job, current stats:\n{stats}")

		# starting actor processes
		for device in self.device_iterator:
			for i in range(self.num_threads):
				actor = ctx.Process(
					target=self.actor,
					args=(
						i,
						device,
						free_queue[device],
						full_queue[device],
						models[device],
						buffers[device]
					))
				actor.start()
				actor_processes.append(actor)

		def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
			"""
			A learner thread starts multiple actors and then batches their data.
			:param i: (int) index of the learner thread
			:param device: (str) device on which the learner thread should run
			:param position: (int) position of the learner thread
			:param local_lock: (threading.Lock) lock for the learner thread
			:param position_lock: (threading.Lock) lock for the position
			:param lock: (threading.Lock) lock for the stats
			"""
			nonlocal frames, stats
			while frames < self.total_frames:
				batch = self.get_batch(
					free_queue=free_queue[device][position],
					full_queue=full_queue[device][position],
					buffers=buffers[device][position],
					lock=local_lock
				)
				_stats = self.learn(
					position=position,
					actor_models=models,
					learn_model=learner_model.get_model(position),
					batch=batch,
					optimizer=optimizers[position],
					position_lock=position_lock
				)
				with lock:
					for k in _stats:
						stats[k] = _stats[k]
					to_log = dict(frames=frames)
					to_log.update({k: stats[k] for k in stat_keys})
					self.file_log.log(to_log)
					frames += self.unroll_length * self.batch_size

		for device in self.device_iterator:
			for m in range(self.num_buffers):
				for position in self.positions:
					free_queue[device][position].put(m)

		locks = {}
		threads = []
		position_locks = {position: threading.Lock() for position in self.positions}
		for device in self.device_iterator:
			locks[device] = {position: threading.Lock() for position in self.positions}
		for device in self.device_iterator:
			for i in range(self.num_threads):
				for position in self.positions:
					thread = threading.Thread(
						target=batch_and_learn,
						name='batch-and-learn-%d' % i,
						args=(
							i,
							device,
							position,
							locks[device][position],
							position_locks[position]))
					thread.start()
					threads.append(thread)

		def checkpoint(frames):
			"""
			Save checkpoint.
			:param frames: (int) number of frames since the start of the training
			"""
			Logger.info('Saving checkpoint to %s', self.checkpoint_path)
			_agents = learner_model.get_models()
			torch.save({
				'model_state_dict': [_agent.state_dict() for _agent in _agents.values()],
				'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers.values()],
				# 'scheduler_state_dict': [scheduler.state_dict() for scheduler in schedulers.values()],
				"stats": stats,
				'frames': frames},
				self.checkpoint_path
			)
			for position in self.positions:
				model_weights_dir = os.path.expandvars(os.path.expanduser(
					'%s/%s/%s' % (self.save_dir, self.x_pid, str(position) + '_' + str(frames) + '.pth')))
				torch.save(learner_model.get_model(position), model_weights_dir)

		timer = timeit.default_timer
		try:
			last_checkpoint_time = timer() - self.save_interval * 60
			while frames < self.total_frames:
				start_frames = frames
				start_time = timer()
				time.sleep(5)
				if timer() - last_checkpoint_time > self.save_interval * 60:
					checkpoint(frames)
					last_checkpoint_time = timer()
				end_time = timer()
				fps = (frames - start_frames) / (end_time - start_time)
				if fps != 0.0:
					Logger.info(
						'After %i frames: @ %.1f fps Stats:\n%s',
						frames,
						fps,
						pprint.pformat(stats))
		except KeyboardInterrupt:
			return
		else:
			for thread in threads:
				thread.join()
			Logger.info('Learning finished after %d frames.', frames)
		checkpoint(frames)
		self.file_log.close()

	def actor(self, i, device, free_queue, full_queue, model, buffers):
		"""
		Actor process which interacts with the environment and preprocesses the data from the environment.
		:param i: (int) the id of the actor
		:param device: (torch.device) the device type
		:param free_queue: (mp.SimpleQueue) a queue to put the buffer when it's free
		:param full_queue: (mp.SimpleQueue) a queue to get a buffer when it's full
		:param model: (torch.nn.Module) the model
		:param buffers: (dict) a dictionary of buffers
		"""
		try:
			Logger.info('Device %s Actor %i started.', str(device), i)
			done_buf = {p: [] for p in self.positions}
			episode_return_buf = {p: [] for p in self.positions}
			target_buf = {p: [] for p in self.positions}
			obs_x_no_action_buf = {p: [] for p in self.positions}
			obs_action_buf = {p: [] for p in self.positions}
			obs_z_buf = {p: [] for p in self.positions}
			size = {p: 0 for p in self.positions}

			# Initialize the training environment
			position, obs, env_output = self.env.initial()

			while True:
				while True:
					obs_z_buf[position].append(env_output['obs_z'])
					obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
					with torch.no_grad():
						action = model.predict(obs=obs, z=obs['z_batch'], x=obs['x_batch'], position=position)
					obs_action_buf[position].append(self.env.actions2tensor(action))
					size[position] += 1
					# Send the buffer to the learner from game
					position, obs, env_output = self.env.step(action)
					if env_output['done']:
						for p in self.positions:
							size_diff = size[p] - len(target_buf[p])
							if size_diff > 0:
								done_buf[p].extend([False for _ in range(size_diff - 1)])
								done_buf[p].append(True)
								episode_return = env_output['episode_return'][p]
								episode_return_buf[p].extend([0.0 for _ in range(size_diff - 1)])
								episode_return_buf[p].append(episode_return)
								target_buf[p].extend([episode_return for _ in range(size_diff)])
						break
				for p in self.positions:
					while size[p] > self.unroll_length:
						index = free_queue[p].get()
						if index is None:
							break
						for t in range(self.unroll_length):
							buffers[p]['done'][index][t, ...] = done_buf[p][t]
							buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
							buffers[p]['target'][index][t, ...] = target_buf[p][t]
							buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
							buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
							buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
						full_queue[p].put(index)
						done_buf[p] = done_buf[p][self.unroll_length:]
						episode_return_buf[p] = episode_return_buf[p][self.unroll_length:]
						target_buf[p] = target_buf[p][self.unroll_length:]
						obs_x_no_action_buf[p] = obs_x_no_action_buf[p][self.unroll_length:]
						obs_action_buf[p] = obs_action_buf[p][self.unroll_length:]
						obs_z_buf[p] = obs_z_buf[p][self.unroll_length:]
						size[p] -= self.unroll_length
		except KeyboardInterrupt:
			pass
		except Exception as e:
			Logger.error('Exception in worker process %i', i)
			traceback.print_exc()
			print()
			raise e