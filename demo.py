import torch

x = torch.tensor([[0.2353, 0.3338, 0.2346, 0.1148, 0.0815]])
print(x.shape)
print(x[0].shape)
print(x[0])
print(x[:].shape)
print(x[:])
