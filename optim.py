import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()

# 2. 创建优化器实例
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练循环
for epoch in range(epochs):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()  # 清空梯度缓存，避免梯度累积
    loss.backward()        # 计算梯度

    # 参数更新
    optimizer.step()       # 更新参数