import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from .dinov2_wrapper import DINOv2Wrapper

class LoRAModel(nn.Module):
    """DINOv2 with LoRA adaptation"""
    
    def __init__(self, model_name="dinov2_vitb14", num_labels=14, 
                 lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        
        # Load base model
        self.backbone = DINOv2Wrapper(model_name, freeze=False)
        
        # LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],  # Target attention layers
            lora_dropout=lora_dropout,
            bias="none",
        )
        
        # Apply LoRA
        self.backbone.dinov2 = get_peft_model(self.backbone.dinov2, lora_config)
        
        # Classifier head
        self.classifier = nn.Linear(self.backbone.feature_dim, num_labels)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
