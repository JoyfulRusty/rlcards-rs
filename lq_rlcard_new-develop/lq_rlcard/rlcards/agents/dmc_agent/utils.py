# -*- coding: utf-8 -*-

import torch
import logging
import traceback

from rlcards.agents.dmc_agent.lookahead import LookAhead


handle = logging.StreamHandler()
handle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s]' '%(message)s'
    )
)

log = logging.getLogger('monster')
log.propagate = False
log.addHandler(handle)
log.setLevel(logging.INFO)

def get_batch(free_queue, full_queue, buffers, batch_size, lock):
    """
    批次数据
    """
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
    batch = {key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers}
    for m in indices:
        free_queue.put(m)
    return batch

def create_buffers(T, num_buffers, state_shape, action_shape, device_iterator):
    """
    创建缓存池
    """
    buffers = {}
    dyg_conv_shape = [[4, 95], [4, 95], [4, 95], [4, 95]]
    mahjong_conv_shape = [[11, 107], [11, 107], [11, 107], [11, 107]]
    sy_tx_conv_shape = [[9, 381], [9, 381], [9, 381], [9, 381]]
    sy_tx_gz_conv_shape = [[9, 427], [9, 427], [9, 427], [9, 427]]
    new_gz_conv_shape = [[16, 115], [16, 115], [16, 115], [16, 115]]

    for device in device_iterator:
        buffers[device] = []
        for player_id in range(len(state_shape)):
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                state=dict(size=(T,) + tuple(sy_tx_gz_conv_shape[player_id]), dtype=torch.float32),
                action=dict(size=(T,) + tuple(action_shape[player_id]), dtype=torch.int8))
            _buffers = {key: [] for key in specs}
            for _ in range(num_buffers):
                for key in _buffers:
                    if device == "cpu":
                        _buffer = torch.empty(**specs[key]).to('cpu').share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to('cuda:' + str(device)).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device].append(_buffers)
    return buffers

def create_optimizers(num_players, learning_rate, learner_model, momentum, eps, alpha):
    """
    创建优化器
    """
    optimizers = []
    for player_id in range(num_players):
        base_optimizer = torch.optim.RMSprop(
            learner_model.parameters(player_id),
            lr=learning_rate,
            momentum=momentum,
            eps=eps,
            alpha=alpha
        )
        optimizer = LookAhead(base_optimizer, k=6, alpha=0.5)
        optimizers.append(optimizer)
    return optimizers

def act(i, device, T, free_queue, full_queue, model, buffers, env):
    try:
        log.info('Device %s Actor %i started.', str(device), i)
        env.seed(i)
        env.set_agents(model.get_agents())

        # 初始化存储参数
        done_buf = [[] for _ in range(env.num_players)]
        episode_return_buf = [[] for _ in range(env.num_players)]
        target_buf = [[] for _ in range(env.num_players)]
        state_buf = [[] for _ in range(env.num_players)]
        action_buf = [[] for _ in range(env.num_players)]
        size = [0 for _ in range(env.num_players)]

        while True:
            trajectories, payoffs, _, _ = env.run(is_training=True)
            for p in range(env.num_players):
                size[p] += len(trajectories[p][:-1]) // 2
                diff = size[p] - len(target_buf[p])
                if diff > 0:
                    done_buf[p].extend([False for _ in range(diff - 1)])
                    done_buf[p].append(True)
                    episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                    episode_return_buf[p].append(float(payoffs[p]))
                    target_buf[p].extend([float(payoffs[p]) for _ in range(diff)])
                    for i in range(0, len(trajectories[p]) - 2, 2):
                        # 卷积残差网络
                        state = trajectories[p][i]['z_obs']
                        # 长短期记忆网络
                        # state = trajectories[p][i]['obs']
                        # 动作特征编码
                        action = env.get_action_feature(trajectories[p][i + 1])
                        state_buf[p].append(torch.from_numpy(state))
                        action_buf[p].append(torch.from_numpy(action))

                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    # [t, ...]: a[:, :, None]和a[..., None]的输出是一样的，就是因为...代替了前面两个冒号
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['state'][index][t, ...] = state_buf[p][t]
                        buffers[p]['action'][index][t, ...] = action_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    state_buf[p] = state_buf[p][T:]
                    action_buf[p] = action_buf[p][T:]
                    size[p] -= T
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def count_no(winner_counts, count_win):
    """
    计算不复活
    """
    print("输出所有对局结果: ", winner_counts)
    if winner_counts["farmer"] > 15:
        winner_counts["farmer"] = 0
    if winner != "farmer":
        count_win[winner_counts["farmer"]] += 1
        winner_counts["farmer"] = 0
        print("输出当前{}对局情况: {}".format("farmer", count_win))

    return winner_counts, count_win

def count_has(winner, winner_counts, count_win):
    """
    计算复活
    """
    if winner != "landlord" and winner_counts["landlord"] < 16:
        count_win[winner_counts["landlord"]] += 1
        winner_counts["landlord"] = 0
        print("更新统计后所有对局结果count_win: ", count_win)
    # 设置闯关结束
    if winner != "landlord":
        farmer_list.append(winner_counts["landlord"])
        print("更新并输出当前闲家连续赢了{}局~".format(winner_counts["landlord"]))
        if winner_counts["landlord"] > 15:
            farmer_list = []
            winner_counts["landlord"] = 0
        if len(farmer_list) == 1:
            winner_counts["landlord"] = 0
        if 1 in set(farmer_list) and len(farmer_list) == 2:
            count_win[winner_counts["landlord"]] += 1
            farmer_list = []
            winner_counts["landlord"] = 0
        elif winner_counts["landlord"] != 1 and winner_counts["landlord"] < 15:
            count_win[winner_counts["landlord"]] += 1
            farmer_list = []
            winner_counts["landlord"] = 0

        print("更新统计后所有对局结果: ", winner_counts)
    elif winner != "landlord":
        print("更新并输出当前庄家赢了{}局~".format(winner_counts[winner]))
        winner_counts[winner] = 0
        print("====================当前更新完毕====================")
        print()
    print("打印第{}局，当前玩家分别[1 ~ 10]赢的次数统计: {}".format(step_count, count_win))
    print()

    return winner_counts, count_win

def gap_winner(winner, winner_counts, count_win, count_roles):
    """
    统计间隔次数的赢家
    """
    if winner != "landlord":
        return

def test_act(i, device, T, free_queue, full_queue, model, buffers, env):
    try:
        log.info('Device %s Actor %i started.', str(device), i)

        # 读取游戏模型基础环境
        env.seed(i)
        env.set_agents()

        winner_counts = {
            "landlord": 0,
            "farmer": 0,
            "draw": 0
        }

        step_count = 0
        farmer_list = []
        count_farmer = []
        step_landlord = 0
        count_landlord = []
        count_win = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        while True:
            winner = env.run()
            winner_counts[winner] += 1
            step_count += 1
            if winner == "draw":
                continue
            if winner == "landlord" and step_landlord < 2:
                step_count += 1
                count_landlord.append(winner_counts[winner])
            # if winner != "landlord" and winner_counts["landlord"] < 16:
            #     count_win[winner_counts["landlord"]] += 1
            #     winner_counts["landlord"] = 0
            #     print("更新统计后所有对局结果count_win: ", count_win)
            # 设置闯关结束
            if winner != "landlord":
                farmer_list.append(winner_counts["landlord"])
                print("更新并输出当前闲家连续赢了{}局~".format(winner_counts["landlord"]))
                if winner_counts["landlord"] > 15:
                    farmer_list = []
                    winner_counts["landlord"] = 0
                    continue
                if 1 in set(farmer_list) and len(farmer_list) == 2:
                    count_win[winner_counts["landlord"]] += 1
                    farmer_list = []
                    winner_counts["landlord"] = 0
                elif winner_counts["landlord"] != 1 and winner_counts["landlord"] < 15:
                    count_win[winner_counts["landlord"]] += 1
                    farmer_list = []
                    winner_counts["landlord"] = 0

                print("更新统计后所有对局结果: ", winner_counts)
            elif winner != "landlord":
                print("更新并输出当前庄家赢了{}局~".format(winner_counts[winner]))
                winner_counts[winner] = 0
                print("====================当前更新完毕====================")
                print()
            print("打印第{}局，当前玩家分别[1 ~ 10]赢的次数统计: {}".format(step_count, count_win))
            print()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e