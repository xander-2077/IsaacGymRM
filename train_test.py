from env import Soccer
from isaacgym import gymapi
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
parser.add_argument('--num_env', default=1, type=int)
parser.add_argument('--num_agent', type=int, default=2)
parser.add_argument('--headless', default=False, action='store_true')

parser.add_argument('--episode_length', type=int, default=500)

args = parser.parse_args()


if __name__ == '__main__':
    env = Soccer(args)
    action_r = torch.randn((args.num_env, args.num_agent * 4), device=args.sim_device) * 5
    action_b = torch.zeros_like(action_r)
    step = 0
    action = torch.randn((args.num_env, args.num_agent * 4 * 2), device=args.sim_device) * 5

    
    while True:
        # env.step((action_r, action_b))
        env.step(action)
        step += 1
        if step % 100 == 0:
            env.reset()