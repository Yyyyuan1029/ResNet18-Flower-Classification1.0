import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    """
    ResNet18 model for flower classification with transfer learning
    """
    
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=True):
        """
        Initialize ResNet18 classifier
        
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: Freeze convolutional layers for transfer learning
        """
        super(ResNet18Classifier, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Initialize the new layer
        nn.init.xavier_uniform_(self.resnet.fc.weight)
        if self.resnet.fc.bias is not None:
            nn.init.constant_(self.resnet.fc.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        return self.resnet(x)
    
    def unfreeze_layers(self, num_layers=2):
        """
        Unfreeze the last few layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        # Get all layers
        layers = list(self.resnet.children())
        
        # Unfreeze the last num_layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always unfreeze the final fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer"""
        return filter(lambda p: p.requires_grad, self.parameters())