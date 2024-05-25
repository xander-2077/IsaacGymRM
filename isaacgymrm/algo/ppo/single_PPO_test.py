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

from env import Soccer
from utils.gpu_manager import *


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












if __name__ == '__main__':
    main()