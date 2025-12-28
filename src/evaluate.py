import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ResNet18Classifier
from src.dataset import get_dataloaders
from src.utils import load_checkpoint, plot_confusion_matrix, print_classification_report, calculate_metrics

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_labels = []
    all_outputs = []
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            
            # Calculate accuracy
            accuracy, predicted = calculate_metrics(outputs, labels)
            total_correct += (predicted == labels).sum().item()
            
            # Store for later analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    # Calculate overall metrics
    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    
    return avg_loss, accuracy, all_predictions, all_labels, all_outputs

def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet18 model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/synthetic', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    _, test_loader, class_names = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Initialize model
    num_classes = len(class_names)
    model = ResNet18Classifier(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    epoch, history = load_checkpoint(str(checkpoint_path), model)
    
    # Evaluate model
    loss, accuracy, predictions, labels, outputs = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Number of test samples: {len(labels)}")
    
    # Generate classification report
    report_df = print_classification_report(labels, predictions, class_names)
    
    # Plot confusion matrix
    cm_path = output_path / 'confusion_matrix.png'
    cm = plot_confusion_matrix(labels, predictions, class_names, save_path=cm_path)
    
    # Save evaluation results
    results = {
        'checkpoint': str(checkpoint_path),
        'test_loss': float(loss),
        'test_accuracy': float(accuracy),
        'num_test_samples': len(labels),
        'class_names': class_names,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_df.to_dict()
    }
    
    results_path = output_path / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*50)

if __name__ == '__main__':
    main()