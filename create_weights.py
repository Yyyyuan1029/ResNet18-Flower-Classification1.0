# create_weights.py
import torch
import os

def create_example_weights():
    """创建示例权重文件"""
    
    # 确保checkpoints目录存在
    os.makedirs('checkpoints', exist_ok=True)
    
    # 创建一个小的示例模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 3)
        
        def forward(self, x):
            return self.layer(x)
    
    model = SimpleModel()
    
    # 保存权重
    save_path = 'checkpoints/model_weights_example.pth'
    torch.save(model.state_dict(), save_path)
    
    # 检查文件大小
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"示例权重文件已创建: {save_path}")
    print(f"文件大小: {file_size:.2f} KB")
    
    return save_path

if __name__ == "__main__":
    create_example_weights()