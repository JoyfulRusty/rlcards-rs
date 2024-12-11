import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_buffers, create_optimizers, act

mean_episode_return_buf = {p: deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}

def compute_loss(log_its, targets):
    loss = ((log_its.squeeze(-1) - targets)**2).mean()
    return loss

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """
    执行学习(优化)步骤
    """
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
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

    # 上锁🔒
    with lock:
        # 模型输出
        learner_outputs = model(obs_z, obs_x, return_value=True)
        # 计算损失值
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_'+position:
                torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+position:
                loss.item(),
        }
        # 梯度归零
        optimizer.zero_grad()
        # 反向传递
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        #保存状态字典
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train(flags):  
    """
    这是训练的主要功能，它将首先初始化所有内容，如缓冲区、优化器等。
    然后，它将作为参与者启动子流程，它会唤醒具有多个线程的学习函数
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please dmc with CPU with `python3 dmc_resnet.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), \
            'The number of actor devices can not exceed the number of available devices'

    # 初始化评估模型
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # 初始化经验缓存池
    buffers = create_buffers(flags, device_iterator)

    # 初始化队列
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queues = {}
    full_queues = {}

    for device in device_iterator:
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queues[device] = _free_queue
        full_queues[device] = _full_queue

    # 创建训练模型
    learner_model = Model(device=flags.training_device)

    # 创建优化器
    optimizers = create_optimizers(flags, learner_model)

    # 座位对应的值
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}

    # 加载模型检查点
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # 启动参与者进程
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queues[device], full_queues[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """
        学习过程的线程目标
        """
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(
                free_queues[device][position],
                full_queues[device][position],
                buffers[device][position],
                flags,
                local_lock
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
            free_queues[device]['landlord'].put(m)
            free_queues[device]['landlord_up'].put(m)
            free_queues[device]['landlord_down'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(
                        i, device, position, locks[device][position], position_locks[position])
                )
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # 保存权重用于评估
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
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

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
