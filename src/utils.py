import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def save_checkpoint(state, filename):
    """
    Save model checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('history', {})

def plot_training_curve(history, save_path=None):
    """
    Plot training and validation curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm

def print_classification_report(y_true, y_pred, class_names):
    """
    Print classification report
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)
    
    # Convert to DataFrame for saving
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    
    return df

def calculate_metrics(outputs, labels):
    """
    Calculate accuracy metrics
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    return accuracy, predicted

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_metrics(metrics, filepath):
    """
    Save metrics to JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")