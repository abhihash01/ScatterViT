import torch
from torch.utils.data import DataLoader
import yaml
from src.models.peft_model import LoRAModel
from src.data.preprocessing import XRayDataset
from src.utils.losses import FocalLoss
from src.utils.metrics import MultiLabelMetrics




class PEFTTrainer:
    def __init__(self, config_path="../config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
    


    def setup_model(self):
        self.model = LoRAModel(
            model_name=self.config['model']['dinov2_variant'],
            num_labels=self.config['data']['num_labels'],
            lora_r=self.config['peft']['lora_r'],
            lora_alpha=self.config['peft']['lora_alpha'],
            lora_dropout=self.config['peft']['lora_dropout']
        ).to(self.device)
        
        self.criterion = FocalLoss(gamma=2)
        


        # training only lora parameters and classifier
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'lora' in name or 'classifier' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.config['training']['lr'])
        
    
    
    
    def setup_data(self):
        # annotated data for fine tuning
        train_paths = ['../data/annotated/real/train/images']  
        train_labels = torch.rand(len(train_paths), self.config['data']['num_labels'])  # labels need to be read according to the data format
        
        val_paths = ['../data/annotated/real/val/images']  
        val_labels = torch.rand(len(val_paths), self.config['data']['num_labels'])  # labels need to be read according to the data format
        
        self.train_dataset = XRayDataset(train_paths, train_labels, augment=True)
        self.val_dataset = XRayDataset(val_paths, val_labels, augment=False)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False)
    
    
    
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    
    
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                all_preds.append(torch.sigmoid(output))
                all_targets.append(target)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = MultiLabelMetrics.compute_metrics(all_targets, all_preds)
        return total_loss / len(self.val_loader), metrics
    
    
    
    
    def train(self):
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["training"]["epochs"]}')
            
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Metrics: {metrics}')
            
            # Early stopping
            if metrics['macro_f1'] > best_f1:
                best_f1 = metrics['macro_f1']
                torch.save(self.model.state_dict(), 'best_peft_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['patience']:
                    print("Early stopping triggered!")
                    break

if __name__ == "__main__":
    trainer = PEFTTrainer()
    trainer.train()
