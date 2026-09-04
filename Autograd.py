import torch

# 构造训练数据：y = 2x + 1 加噪声
torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# 初始化模型参数（需要追踪梯度）
w = torch.zeros(1, requires_grad=True)   # 权重
b = torch.zeros(1, requires_grad=True)   # 偏置

lr = 0.1    # 学习率
epochs = 50  # 训练轮数

for epoch in range(epochs):
    # 1. 前向传播：计算预测值
    y_pred = X * w + b

    # 2. 计算损失（均方误差）
    loss = ((y_pred - y_true) ** 2).mean()

    # 3. 反向传播：自动计算 d(loss)/dw 和 d(loss)/db
    loss.backward()

    # 4. 手动更新参数（用 no_grad 包裹，避免更新操作被追踪进计算图）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 5. 清零梯度（下一轮 backward 前必须清零）
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | w={w.item():.3f}, b={b.item():.3f}")

print(f"\n训练完成：w ≈ {w.item():.3f}（真实值 2.0），b ≈ {b.item():.3f}（真实值 1.0）")