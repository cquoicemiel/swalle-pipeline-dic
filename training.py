import torch

data = [[1, 2, 3], [3, 4, 5], [7, 8, 9]]

x_data = torch.tensor(data)

x_rand = torch.rand_like(x_data, dtype=torch.float)

print(x_rand)