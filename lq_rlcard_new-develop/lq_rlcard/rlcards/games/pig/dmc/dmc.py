# -*- coding: utf-8 -*-

import os
import time
import torch
import timeit
import pprint
import threading
import numpy as np

from torch import nn
from .models import Model
from collections import deque
from .file_writer import FileWriter
from torch import multiprocessing as mp
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act


ALL_ROLE = ['landlord1', 'landlord2', 'landlord3', 'landlord4']
mean_episode_return_buf = {p: deque(maxlen=100) for p in ALL_ROLE}

def compute_loss(logits, targets):
    """
    计算模型损失值
    todo: 使用squeeze进行维度压缩，大小为1的所有维都删除
    """
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """
    学习优化更新
    """
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_' + position: torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + position: loss.item(),
        }

        optimizer.zero_grad()  # 梯度清零
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())  # 加载模型状态字典
        return stats

def train(flags):
    """
    训练
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. "
                "Otherwise, please dmc with CPU with `python3 dmc.py --actor_device_cpu --training_device cpu`"
            )
    plogger = FileWriter(
        x_pid=flags.x_pid,
        xp_args=flags.__dict__,
        root_dir=flags.save_dir,
    )
    checkpoint_path = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.save_dir, flags.x_pid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # 初始化模型
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # 初始化缓存buffer
    buffers = create_buffers(flags, device_iterator)

    # 初始化队列
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = {role: ctx.SimpleQueue() for role in ALL_ROLE}
        _full_queue = {role: ctx.SimpleQueue() for role in ALL_ROLE}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # 训练学习模型
    learner_model = Model(device=flags.training_device)

    # 优化器
    optimizers = create_optimizers(flags, learner_model)

    # 模型 -> 损失
    stat_keys = [
        'mean_episode_return_landlord1',
        'loss_landlord1',
        'mean_episode_return_landlord2',
        'loss_landlord2',
        'mean_episode_return_landlord3',
        'loss_landlord3',
        'mean_episode_return_landlord4',
        'loss_landlord4',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {role: 0 for role in ALL_ROLE}

    # 如果存在已训练模型，则加载检查点，继续训练
    if flags.load_model and os.path.exists(checkpoint_path):
        checkpoint_states = torch.load(
            checkpoint_path,
            map_location=("cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ALL_ROLE:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # 启动进程
    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """
        批次学习
        """
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(
                free_queue[device][position],
                full_queue[device][position],
                buffers[device][position],
                flags, local_lock
            )
            _stats = learn(
                position,
                models,
                learner_model.get_model(position),
                batch,
                optimizers[position],
                flags,
                position_lock
            )
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['landlord1'].put(m)
            free_queue[device]['landlord2'].put(m)
            free_queue[device]['landlord3'].put(m)
            free_queue[device]['landlord4'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {role: threading.Lock() for role in ALL_ROLE}
    position_locks = {role: threading.Lock() for role in ALL_ROLE}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ALL_ROLE:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i,
                    args=(i, device, position, locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpoint_path)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpoint_path)

        # 保存权重评估
        for position in ALL_ROLE:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.save_dir, flags.x_pid, position + '_weights_' + str(frames) + '.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info(
                'After %i (L1:%i L2:%i L3:%i L4:%i) frames: @ %.1f fps (avg@ %.1f fps) (L1:%.1f L2:%.1f L3:%.1f L4:%.1f) Stats:\n%s',
                frames,
                position_frames['landlord1'],
                position_frames['landlord2'],
                position_frames['landlord3'],
                position_frames['landlord4'],
                fps,
                fps_avg,
                position_fps['landlord1'],
                position_fps['landlord2'],
                position_fps['landlord3'],
                position_fps['landlord4'],
                pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    checkpoint(frames)
    plogger.close()
