import torch.nn as nn
from torchvision import models

class TumorClassifier(nn.Module):
    """ResNet50-based tumor classifier with custom head"""
    def __init__(self, num_classes=1):
        super(TumorClassifier, self).__init__()
        # Load pretrained ResNet50
        self.mri_classifier = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.mri_classifier.parameters())[:-6]:
            param.requires_grad = False
            
        # Modify the final layer for binary classification
        num_features = self.mri_classifier.fc.in_features
        self.mri_classifier.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.mri_classifier(x)