# -*- coding: utf-8 -*-

import os
import time
import torch
import timeit
import pprint
import random
import threading
import numpy as np


from collections import deque
from torch import multiprocessing as mp, nn

from reinforce.model.dyg_conv import DMCDygModel as DygModel
from reinforce.dmc_v1.file_writer import FileWriter

from reinforce.dmc_v1.utils import (
    get_batch,
    create_buffers,
    create_optimizers,
    act,
    log,
)


def compute_loss(output_vals, targets, device="cpu"):
    """
    todo: 计算训练损失值
    """
    loss_func = torch.nn.MSELoss()
    return loss_func(output_vals.squeeze(-1).to(device), targets.to(device))

def learn(
        position,
        actor_models,
        agent,
        batch,
        optimizer,
        training_device,
        max_grad_norm,
        mean_episode_return_buf,
        lock):
    """
    执行学习(优化)步骤
    """
    with torch.no_grad():
        device = "cuda:" + str(training_device) if training_device != "cpu" else "cpu"
        state = torch.flatten(batch['state'].to(device), 0, 1).float()
        action = torch.flatten(batch['action'].to(device), 0, 1).float()
        target = torch.flatten(batch['target'].to(device), 0, 1)
        episode_returns = batch['episode_return'][batch['done']]
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    # 加锁
    with lock:
        # 前向传递
        values = agent.forward(state, action)
        # 计算损失
        loss = compute_loss(values, target)
        # 玩家座位对应奖励和损失值
        stats = {
            'mean_episode_return_' + str(position):
                torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + str(position): loss.item(),
        }
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        for actor_model in actor_models.values():
            actor_model.get_agent(position).load_state_dict(agent.state_dict())
        return stats

class DMCTrainer:
    """
    深度蒙特卡洛
    """
    def __init__(
            self,
            env,
            cuda="",
            load_model=False,
            x_pid='dmc_v1',
            save_interval=30,
            num_actor_devices=1,
            num_actors=5,
            training_device="0",
            save_dir='result/dmc_v1',
            total_frames=100000000000,
            exp_epsilon=0.01,
            batch_size=32,
            unroll_length=100,
            num_buffers=50,
            num_threads=4,
            max_grad_norm=40,
            learning_rate=0.0001,
            alpha=0.99,
            momentum=0.40,
            epsilon=0.00001):

        # 环境
        self.env = env

        # 日志
        self.p_logger = FileWriter(
            x_pid=x_pid,
            root_dir=save_dir,
        )

        # 检查点路径
        self.checkpoint_path = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (save_dir, x_pid, 'model_1.tar')))

        self.T = unroll_length
        self.B = batch_size

        self.x_pid = x_pid
        self.alpha = alpha
        self.save_dir = save_dir
        self.epsilon = epsilon
        self.momentum = momentum
        self.num_actors = num_actors
        self.load_model = load_model
        self.exp_epsilon = exp_epsilon
        self.num_buffers = num_buffers
        self.num_threads = num_threads
        self.total_frames = total_frames
        self.save_interval = save_interval
        self.num_actor_devices = num_actor_devices
        self.training_device = training_device
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        self.num_players = self.env.num_players
        self.action_shape = self.env.action_shape
        # One-hot 编码
        if not self.action_shape[0]:
            self.action_shape = [[self.env.num_actions] for _ in range(self.num_players)]

        def model_func(device):
            return DygModel(
                self.env.state_shape,
                self.action_shape,
                exp_epsilon=self.exp_epsilon,
                device=str(device),
            )

        self.model_func = model_func

        # mean_episode_return_buf
        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.num_players)]

        if cuda == "":  # Use CPU
            self.device_iterator = ["cpu"]
            self.training_device = "cpu"
        else:
            self.device_iterator = range(num_actor_devices)

    @staticmethod
    def seed_torch():
        random.seed(1024)  # Python的随机性
        np.random.seed(1024)  # numpy的随机性
        torch.manual_seed(1024)  # torch的CPU随机性，为CPU设置随机种子
        torch.cuda.manual_seed(1024)  # torch的GPU随机性，为当前GPU设置随机种子
        torch.cuda.manual_seed_all(1024)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子

    def start(self):
        # 初始化动作模型
        models = {}
        for device in self.device_iterator:
            model = self.model_func(device)
            model.share_memory()
            model.eval()
            models[device] = model

        # 初始化缓冲区
        buffers = create_buffers(
            self.T,
            self.num_buffers,
            self.env.state_shape,
            self.action_shape,
            self.device_iterator
        )

        # 初始化队列
        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = {}
        full_queue = {}
        for device in self.device_iterator:
            _free_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
            _full_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
            free_queue[device] = _free_queue
            full_queue[device] = _full_queue

        # 训练学习者模型
        learner_model = self.model_func(self.training_device)

        # 创建优化器
        optimizers = create_optimizers(
            self.num_players,
            self.learning_rate,
            learner_model,
            self.momentum,
            self.epsilon,
            self.alpha
        )

        # 统计键
        stat_keys = []
        for p in range(self.num_players):
            stat_keys.append('mean_episode_return_' + str(p))
            stat_keys.append('loss_' + str(p))
        frames, stats = 0, {k: 0 for k in stat_keys}

        # 当存在检查点时，加载检查点
        if self.load_model and os.path.exists(self.checkpoint_path):
            # 检查点状态
            checkpoint_states = torch.load(
                self.checkpoint_path,
                map_location="cuda:" + str(self.training_device) if self.training_device != "cpu" else "cpu")

            # 从检查点状态获取模型状态字典到学习模型中，从检查点状态中获取模型优化器状态字典到优化器中
            for p in range(self.num_players):
                learner_model.get_agent(p).load_state_dict(checkpoint_states["model_state_dict"][p])
                optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
                # 运行设备
                for device in self.device_iterator:
                    models[device].get_agent(p).load_state_dict(learner_model.get_agent(p).state_dict())
            stats = checkpoint_states["stats"]
            frames = checkpoint_states["frames"]
            log.info(f"Resuming preempted job, current stats:\n{stats}")

        # 启动动作进程
        for device in self.device_iterator:
            for i in range(self.num_actors):
                actor = ctx.Process(
                    target=act,
                    args=(
                        i,
                        device,
                        self.T,
                        free_queue[device],
                        full_queue[device],
                        models[device],
                        buffers[device],
                        self.env))
                actor.start()
                actor_processes.append(actor)

        def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
            """ 开始学习 """
            nonlocal frames, stats
            while frames < self.total_frames:
                batch = get_batch(
                    free_queue[device][position],
                    full_queue[device][position],
                    buffers[device][position],
                    self.B,
                    local_lock)
                # 学习流程
                _stats = learn(
                    position,
                    models,
                    learner_model.get_agent(position),
                    batch,
                    optimizers[position],
                    self.training_device,
                    self.max_grad_norm,
                    self.mean_episode_return_buf,
                    position_lock)

                with lock:
                    for k in _stats:
                        stats[k] = _stats[k]
                    to_log = dict(frames=frames)
                    to_log.update({k: stats[k] for k in stat_keys})
                    self.p_logger.log(to_log)
                    frames += self.T * self.B

        for device in self.device_iterator:
            for m in range(self.num_buffers):
                for p in range(self.num_players):
                    free_queue[device][p].put(m)

        threads = []
        locks = {device: [threading.Lock() for _ in range(self.num_players)] for device in self.device_iterator}
        position_locks = [threading.Lock() for _ in range(self.num_players)]

        for device in self.device_iterator:
            for i in range(self.num_threads):
                for position in range(self.num_players):
                    thread = threading.Thread(
                        target=batch_and_learn,
                        name='batch-and-learn-%d' % i,
                        args=(
                            i,
                            device,
                            position,
                            locks[device][position],
                            position_locks[position]))

                    # 线程开启
                    thread.start()
                    threads.append(thread)

        def checkpoint(frames):
            """
            检查点
            """
            log.info('Saving checkpoint to %s', self.checkpoint_path)
            _agents = learner_model.get_agents()
            torch.save({
                'model_state_dict': [_agent.state_dict() for _agent in _agents],
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                "stats": stats,
                'frames': frames},
                self.checkpoint_path)
            # 保存权重用于评估
            for position in range(self.num_players):
                model_weights_dir = os.path.expandvars(os.path.expanduser(
                    '%s/%s/%s' % (self.save_dir, self.x_pid, str(position) + '_' + str(frames) + '.pth')))
                torch.save(
                    learner_model.get_agent(position),
                    model_weights_dir)
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
                    log.info(
                        'After %i frames: @ %.1f fps Stats:\n%s',
                        frames,
                        fps,
                        pprint.pformat(stats))
        except KeyboardInterrupt:
            return
        else:
            for thread in threads:
                thread.join()
            log.info('Learning finished after %d frames.', frames)

        checkpoint(frames)
        self.p_logger.close()