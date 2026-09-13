import torch
import torch.nn as nn

tanh = nn.Tanh()

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tanh(x)

print("输入:", x.tolist())
print("输出:", output.tolist())
print("特点: 输出范围 [-1, 1]，零中心化")