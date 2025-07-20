import torch
import torch.nn as nn
from .dinov2_wrapper import DINOv2Wrapper

class LinearProbe(nn.Module):
    """Linear probe classifier on DINOv2 features"""
    
    def __init__(self, model_name="dinov2_vitb14", num_labels=14, dropout=0.1):
        super().__init__()
        self.backbone = DINOv2Wrapper(model_name, freeze=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, num_labels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class MLPProbe(nn.Module):
    """MLP probe classifier on DINOv2 features"""
    
    def __init__(self, model_name="dinov2_vitb14", num_labels=14, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.backbone = DINOv2Wrapper(model_name, freeze=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
