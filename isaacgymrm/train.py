from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import os
import wandb
import hydra
import setproctitle
import numpy as np
from pathlib import Path
import math
import torch

from isaacgymrm.env.env import Soccer
from utils.gpu_manager import *

from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG

@hydra.main(config_path="cfg", config_name="config", version_base="1.3")
def main(cfg):

    # Cuda
    gpu_manager = GPUManager()
    device = torch.device(type='cuda', index=gpu_manager.auto_choice()[0])
    
    torch.set_num_threads(cfg.n_training_threads)
    if cfg.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    env = Soccer(cfg)
    step = 0

    # # random test
    # while True:
    #     action = torch.randn((cfg.num_env, cfg.num_agent * 4 * 2), device=cfg.sim_device) * 5
    #     env.step(action)
    #     step += 1
    #     if step % 100 == 0:
    #         env.reset()

    # mecanum test
    action = torch.randn((cfg.num_env, cfg.num_agent * 2, 3), device=cfg.sim_device) * 10
    action[..., 0] = 0
    action[..., 1] = 0
    action[..., 2] = 2

    while True:
        env.step(action)
        step += 1
        if step % 100 == 0:
            env.reset()



if __name__ == '__main__':
    main()