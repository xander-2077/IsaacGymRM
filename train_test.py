from env import Soccer
import torch
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument(
    '--sim_device',
    type=str,
    default="cuda:0",
    help='Physics Device in PyTorch-like syntax',
)
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument(
    '--graphics_device_id', type=int, default=0, help='Graphics Device ID'
)
parser.add_argument('--num_envs', default=16, type=int)
parser.add_argument('--num_agent', type=int, default=2)
parser.add_argument('--headless', default=True, action='store_true')

parser.add_argument('--episode_length', type=int, default=500)

args = parser.parse_args()


if __name__ == '__main__':
    env = Soccer(args)
    # import pdb; pdb.set_trace()
    while True:
        action = torch.randint(0, 9, (args.num_envs * 2, 3, 1), device=args.sim_device)
        env.step([action])
