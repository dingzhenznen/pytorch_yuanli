import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = sigmoid(x)

print("输入:", x.tolist())
print("输出:", output.tolist())