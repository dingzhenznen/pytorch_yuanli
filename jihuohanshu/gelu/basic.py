import torch
import torch.nn as nn

# 创建 GELU 激活层
gelu = nn.GELU()

# 测试输入
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 前向传播
output = gelu(x)

print("输入:", x.tolist())
print("输出:", output.tolist())
print("n观察：负值有轻微激活（非零），正值保持增长")