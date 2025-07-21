import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from src.models.linear_probe import LinearProbe, MLPProbe
from src.data.preprocessing import XRayDataset
from src.utils.metrics import MultiLabelMetrics
from src.utils.losses import FocalLoss

class LinearProbeTrainer:
    def __init__(self, config_path="../config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
    
    def setup_model(self):
        # linear probe
        self.linear_model = LinearProbe(
            model_name=self.config['model']['dinov2_variant'],
            num_labels=self.config['data']['num_labels'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        
        #;mlp
        self.mlp_model = MLPProbe(
            model_name=self.config['model']['dinov2_variant'],
            num_labels=self.config['data']['num_labels'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        self.criterion = FocalLoss(gamma=2)
    
    
    
    def setup_data(self):
        # will need to be changed according to the data format of annotated data
        train_paths = ['../data/annotated/train/images']  
        train_labels = torch.rand(len(train_paths), self.config['data']['num_labels'])  # need to replace with actual label reads
        
        self.train_dataset = XRayDataset(train_paths, train_labels)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
    
    def train_model(self, model, model_name):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['lr'])
        
        model.train()
        for epoch in range(self.config['training']['epochs']):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'{model_name} Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return model
    



    def run(self):
        print("Training Linear Probe...")
        self.linear_model = self.train_model(self.linear_model, "Linear")
        
        print("Training MLP Probe...")
        self.mlp_model = self.train_model(self.mlp_model, "MLP")
        
        # saving to chck points
        torch.save(self.linear_model.state_dict(), '../checkpoints/linear_probe.pth')
        torch.save(self.mlp_model.state_dict(), '../checkpoints/mlp_probe.pth')

if __name__ == "__main__":
    trainer = LinearProbeTrainer()
    trainer.run()
