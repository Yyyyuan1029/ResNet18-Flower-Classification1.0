"""
ResNet18 花卉分类训练脚本
"""

import sys
import os

# 将项目根目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # src目录
parent_dir = os.path.dirname(current_dir)  # 项目根目录
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import ResNet18FlowerClassifier
from src.dataset import get_data_loaders


class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        参数:
            config (dict): 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = ResNet18FlowerClassifier(
            num_classes=config['num_classes'],
            pretrained=config['pretrained'],
            freeze_layers=config['freeze_layers']
        ).to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate']
        )
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # 数据加载器
        print("加载数据集...")
        try:
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
                data_dir=config['data_dir'],
                batch_size=config['batch_size']
            )
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("请确保数据集存在且结构正确")
            raise
        
        # 记录训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [训练]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [验证]')
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def test(self):
        """测试模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='[测试]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        test_acc = 100. * correct / total
        print(f'测试准确率: {test_acc:.2f}%')
        
        return test_acc
    
    def train(self):
        """完整的训练流程"""
        print("开始训练...")
        print("-" * 50)
        start_time = time.time()
        
        best_acc = 0.0
        for epoch in range(self.config['epochs']):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 学习率调整
            self.scheduler.step()
            
            # 打印结果
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%')
            print(f'  验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(f'checkpoints/best_model_acc_{val_acc:.2f}.pth')
                print(f'  ✓ 保存最佳模型，准确率: {val_acc:.2f}%')
            else:
                print(f'  - 当前最佳: {best_acc:.2f}%')
            
            print("-" * 40)
        
        # 保存最终模型
        self.save_model('checkpoints/final_model.pth')
        
        # 测试模型
        print("\n测试模型...")
        test_acc = self.test()
        
        training_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"总时间: {training_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"测试准确率: {test_acc:.2f}%")
        
        return self.history
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        print(f"模型已保存到: {path}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        axes[0].plot(self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('训练和验证损失曲线')
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(self.history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[1].plot(self.history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].set_title('训练和验证准确率曲线')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('docs/images', exist_ok=True)
        plt.savefig('docs/images/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练曲线已保存到: docs/images/training_history.png")


def main():
    """主函数"""
    print("=" * 50)
    print("ResNet18 花卉分类训练")
    print("=" * 50)
    
    # 配置参数
    config = {
        'data_dir': 'data',
        'num_classes': 5,
        'batch_size': 8,  # 根据内存调整，如果内存不足可减小到4
        'epochs': 10,
        'learning_rate': 0.001,
        'pretrained': True,
        'freeze_layers': True  # 冻结预训练层，加快训练
    }
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # 创建训练器并开始训练
    try:
        trainer = Trainer(config)
        history = trainer.train()
        
        # 绘制训练曲线
        trainer.plot_training_history()
        
        return history
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("\n可能的原因:")
        print("1. 数据集不存在或路径不正确")
        print("2. 内存不足，尝试减小batch_size")
        print("3. 缺少依赖包，运行: pip install -r requirements.txt")
        
        # 提供测试数据集的选项
        print("\n建议:")
        print("1. 下载花卉数据集: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition")
        print("2. 或运行 create_test_data.py 创建测试数据集")
        
        return None


if __name__ == "__main__":
    main()