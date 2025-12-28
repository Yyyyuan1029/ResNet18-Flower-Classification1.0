# simplest_train.py
import torch
import torch.nn as nn
import torch.optim as optim

# 创建最简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.fc(x)

# 创建虚拟数据
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 虚拟训练
for epoch in range(3):
    # 虚拟数据
    inputs = torch.randn(4, 10)
    labels = torch.randint(0, 3, (4,))
    
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("训练完成！")