import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


class FlowerDataset(Dataset):
    """花卉数据集类"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化数据集
        
        参数:
            root_dir: 数据根目录
            split: 'train', 'val', 或 'test'
            transform: 数据转换
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        
        # 检查目录是否存在
        if not os.path.exists(self.root_dir):
            print(f"警告: 目录 {self.root_dir} 不存在!")
            print("请确保数据集目录结构为:")
            print("  data/train/class1/")
            print("  data/train/class2/")
            print("  ...")
            print("\n可以运行 create_test_data.py 创建测试数据")
            
            # 创建空数据集用于测试
            self.classes = []
            self.images = []
            self.labels = []
            return
        
        # 获取类别列表
        try:
            self.classes = sorted([
                d for d in os.listdir(self.root_dir) 
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])
        except FileNotFoundError:
            print(f"错误: 找不到目录 {self.root_dir}")
            self.classes = []
            self.images = []
            self.labels = []
            return
        
        if not self.classes:
            print(f"警告: 目录 {self.root_dir} 中没有子目录!")
            print("请确保每个类别都有对应的文件夹")
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 收集所有图片路径和标签
        self.images = []
        self.labels = []
        
        print(f"正在扫描 {self.root_dir}...")
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            img_files = [
                f for f in os.listdir(cls_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            
            for img_name in img_files:
                self.images.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_to_idx[cls])
            
            print(f"  类别 '{cls}': {len(img_files)} 张图片")
        
        print(f"总计: {len(self.images)} 张图片")
        
        if len(self.images) == 0:
            print("警告: 没有找到任何图片!")
            print("请检查:")
            print("  1. 数据集路径是否正确")
            print("  2. 图片文件格式是否为 .jpg, .jpeg, .png")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # 加载图片
            image = Image.open(self.images[idx]).convert('RGB')
        except Exception as e:
            print(f"加载图片失败 {self.images[idx]}: {e}")
            # 创建黑色图片作为替代
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """获取数据转换"""
    
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 验证/测试集的转换
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_data_loaders(data_dir='data', batch_size=32):
    """获取数据加载器"""
    
    train_transform, val_transform = get_transforms()
    
    print("加载训练集...")
    train_dataset = FlowerDataset(data_dir, split='train', transform=train_transform)
    
    print("加载验证集...")
    val_dataset = FlowerDataset(data_dir, split='val', transform=val_transform)
    
    print("加载测试集...")
    test_dataset = FlowerDataset(data_dir, split='test', transform=val_transform)
    
    # 检查数据集是否为空
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("数据集为空! 请确保数据集路径正确")
    
    print(f"训练集: {len(train_dataset)} 张图片, {len(train_dataset.classes)} 个类别")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")
    print(f"类别: {train_dataset.classes}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """测试数据集"""
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)
    
    # 检查一个batch
    images, labels = next(iter(train_loader))
    print(f"Batch 形状: {images.shape}")
    print(f"标签: {labels}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    test_dataset()