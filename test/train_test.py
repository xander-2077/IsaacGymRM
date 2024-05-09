from isaacgym import gymapi
import torch
import random
import argparse
import numpy as np

import sys
sys.path.append('.')
from isaacgymrm.env import Soccer

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

# reward scale
parser.add_argument('--reward_scoring', type=int, default=1000)
parser.add_argument('--reward_conceding', type=int, default=1000)
parser.add_argument('--reward_vel_to_ball', type=int, default=0.05)
parser.add_argument('--reward_vel', type=int, default=0.1)







args = parser.parse_args()


if __name__ == '__main__':
    env = Soccer(args)
    step = 0

    # action_r = torch.randn((args.num_env, args.num_agent * 4), device=args.sim_device) * 5
    # action_b = torch.zeros_like(action_r)
    # action = torch.randn((args.num_env, args.num_agent * 4 * 2), device=args.sim_device) * 5

    # while True:
    #     action = torch.randn((args.num_env, args.num_agent * 4 * 2), device=args.sim_device) * 5
    #     env.step(action)
    #     step += 1
    #     if step % 100 == 0:
    #         env.reset()


    # mecanum test
    action = torch.randn((args.num_env, args.num_agent * 2, 3), device=args.sim_device) * 10
    action[..., 0] = 0
    action[..., 1] = 0
    action[..., 2] = 2

    while True:
        env.step(action)
        step += 1
        if step % 100 == 0:
            env.reset()

    