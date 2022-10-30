import torch
t_cpu = torch.tensor([0.1, 0.2])
print(t_cpu.device)
# cpu

print(type(t_cpu.device))
# <class 'torch.device'>

t_gpu = torch.tensor([0.1, 0.2], device='cuda')
print(t_gpu.device)
# cuda:0

print(type(t_gpu.device))
# <class 'torch.device'>