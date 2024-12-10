# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description='Pytorch 2.x Training AI Frame')

# 常规设置
parser.add_argument("--env", type=str, default='monster', choices=['monster'])
parser.add_argument('--x_pid', default='monster', help='Experiment id (default: monster)')
parser.add_argument('--save_interval', default=120, type=int, help='Time interval (in minutes) at which to save the model')
parser.add_argument('--objective', default='adp', type=str, choices=['adp'], help='Use ADP as reward (default: ADP)')

# 训练设置
parser.add_argument('--num_actor_devices', default=1, type=int, help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=4, type=int, help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str, help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', default=True, action='store_true', help='Load an existing model')
parser.add_argument('--save_dir', default='result/dmc/',  help='Root dir where experiment data will be saved')

# 超参数
parser.add_argument('--total_frames', default=100000000000, type=int, help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.1, type=float, help='The probability for exploration')
parser.add_argument('--exp_decay', default=1., type=float)
parser.add_argument('--batch_size', default=30, type=int, help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int, help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int, help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=1, type=int, help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float, help='Max norm of gradients')

# 优化器设置
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float, help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float, help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-8, type=float,  help='RMSProp epsilon')


flags = parser.parse_args()