import torch

# num_envs = 8
# actors_per_env = 4


# indices_1 = actors_per_env * torch.arange(num_envs, dtype=torch.int32)
# indices_2 = torch.arange(num_envs * actors_per_env, dtype=torch.int32).view(num_envs, actors_per_env)

a = torch.randn((8, 4, 2))
import pdb; pdb.set_trace()

b = torch.randn((a.shape[:-1], 3))



import pdb; pdb.set_trace()