import torch
import torch.nn as nn

class DINOv2Wrapper(nn.Module):
    """Simple DINOv2 wrapper for feature extraction"""
    
    def __init__(self, model_name="dinov2_vitb14", freeze=True):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        
        if freeze:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        self.feature_dim = self.dinov2.embed_dim
    
    def forward(self, x):
        return self.dinov2(x)
