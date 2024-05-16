import torch

a = torch.tensor([0.5, 0.5, 0.5])
b = torch.where(torch.tensor([True, False, True]), torch.ones_like(a), a)

print(b)