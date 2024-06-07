import gymnasium as gym
import numpy as np
import tyro
from pprint import pprint

import sys
sys.path.append('/home/xander/Codes/IsaacGym/IsaacGymRM/')
from isaacgymrm.env.env import RoboMasterEnv
import torch

from isaacgymrm.algo.ppo.PPO_isaacgym_CleanRL import Args, RecordEpisodeStatisticsTorch, Agent


def evaluate(
    args,
    model_path: str,
    eval_episodes: int,
    Model: torch.nn.Module,
    device,
):
    envs = RoboMasterEnv(args)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.state_space

    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs = envs.reset()['states']
    episodic_returns = []

    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(obs)
        next_obs, _, _, infos = envs.step(actions)

        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if "episode" not in info:
        #             continue
        #         print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
        #         episodic_returns += [info["episode"]["r"]]

        obs = next_obs

    return episodic_returns



if __name__ == '__main__':
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.headless = False

    model_path = '/home/xander/Codes/IsaacGym/IsaacGymRM/runs/Robomaster_PPO-cleanRL_05-24_22-22/agent.cleanrl_model'

    evaluate(
        args,
        model_path,
        eval_episodes=10,
        Model=Agent,
        device="cuda:0",
    )