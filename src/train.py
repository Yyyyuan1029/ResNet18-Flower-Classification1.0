import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ResNet18Classifier
from src.dataset import get_dataloaders
from src.utils import save_checkpoint, plot_training_curve

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss/total,
            'acc': 100*correct/total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100*correct/total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on synthetic flower dataset')
    parser.add_argument('--data_dir', type=str, default='./data/synthetic', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, test_loader, class_names = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")
    
    # Initialize model
    model = ResNet18Classifier(num_classes=args.num_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.get_trainable_parameters(), 
        lr=args.lr, 
        momentum=args.momentum
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'class_names': class_names
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_acc': val_acc,
            'history': history,
            'class_names': class_names
        }
        
        save_checkpoint(checkpoint, output_path / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint, output_path / 'best_model.pth')
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    save_checkpoint(checkpoint, output_path / 'final_model.pth')
    
    # Save training history
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curve
    plot_training_curve(history, output_path / 'training_curve.png')
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()