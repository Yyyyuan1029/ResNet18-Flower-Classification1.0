# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_layers=True):
        super(ResNet18FlowerClassifier, self).__init__()
        
        # 加载预训练ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 冻结前几层
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 修改最后一层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def test_model():
    model = ResNet18FlowerClassifier(num_classes=5)
    print("模型结构:")
    print(model)
    
    # 测试随机输入
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    return model

if __name__ == "__main__":
    test_model()