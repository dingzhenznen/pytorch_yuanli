import torch
import torch.nn as nn

model = nn.Linear(4, 1)
sigmoid = nn.Sigmoid()

data = torch.randn(4, 4)
print(data)

logits = model(data)
probabilities = sigmoid(logits)

print("Logits:", logits.squeeze().tolist())
print("概率:", probabilities.squeeze().tolist())
print("预测:", (probabilities > 0.5).squeeze().tolist())
