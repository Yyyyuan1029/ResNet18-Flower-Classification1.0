import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from pathlib import Path

class SyntheticFlowerDataset(Dataset):
    """
    Custom dataset for synthetic flower images
    """
    
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing the dataset
            transform: Transformations to apply
            is_train: Whether this is training data
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Determine subdirectory based on train/test
        subdir = 'train' if is_train else 'test'
        
        # Load dataset using ImageFolder
        self.dataset = datasets.ImageFolder(
            self.data_dir / subdir,
            transform=self.transform
        )
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
    
    def __len__(self):
        """Return number of samples"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get sample by index"""
        return self.dataset[idx]

def get_dataloaders(data_dir, batch_size=16, num_workers=2):
    """
    Create data loaders for training and testing
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, test_loader, class_names
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Testing transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Create datasets
    train_dataset = SyntheticFlowerDataset(data_dir, transform=train_transform, is_train=True)
    test_dataset = SyntheticFlowerDataset(data_dir, transform=test_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes