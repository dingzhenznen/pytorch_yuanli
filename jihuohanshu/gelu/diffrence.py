import torch
import torch.nn as nn

x = torch.linspace(-4, 4, 21)

# 不同激活函数
gelu = nn.GELU()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

print("x       GELU      ReLU      Sigmoid   Tanh")
print("-" * 50)
for i in range(0, 21, 3):
    xi = x[i:i+3]
    print(f"{xi[0]:6.2f} {gelu(xi)[0]:8.4f} {relu(xi)[0]:8.4f} {sigmoid(xi)[0]:8.4f} {tanh(xi)[0]:8.4f}")