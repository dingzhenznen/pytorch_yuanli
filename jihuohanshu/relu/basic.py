import torch
import torch.nn as nn

# 创建 ReLU 激活层
relu = nn.ReLU()

# 创建包含负值的输入张量
input_tensor = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])

# 前向传播
output = relu(input_tensor)

print("输入:n", input_tensor)
print("输出:n", output)