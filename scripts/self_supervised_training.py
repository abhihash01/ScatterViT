import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import yaml
from src.models.dinov2_wrapper import DINOv2Wrapper
from src.data.preprocessing import XRayDataset

class DINOLoss(nn.Module):
    """DINO loss for self-supervised learning"""    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, student_output, teacher_output):
        student_probs = F.softmax(student_output / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_output / self.temperature, dim=-1)
        return -(teacher_probs * torch.log(student_probs + 1e-8)).sum(dim=-1).mean()




class SSLTrainer:
    def __init__(self, config_path="../config/config.yaml", use_fsdp=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fsdp = use_fsdp
        self.setup_model()
        self.setup_data()
    
    
    def setup_model(self):
        # Student teacher distillation

        #student model
        self.student = DINOv2Wrapper(
            model_name=self.config['model']['dinov2_variant'], 
            freeze=False
        )
        

        #teacher model
        self.teacher = DINOv2Wrapper(
            model_name=self.config['model']['dinov2_variant'], 
            freeze=True
        )
        
        
        
        # Copy student weights to teacher
        self.teacher.load_state_dict(self.student.state_dict())
        
        
        ######### needs rework and testing- experimentatl #########
        if self.use_fsdp and torch.cuda.device_count() > 1:
            self.student = FSDP(self.student)
            self.teacher = FSDP(self.teacher)
        ########################################################
        
        
        
        self.student.to(self.device)
        self.teacher.to(self.device)
        
        self.criterion = DINOLoss()
    
    def setup_data(self):
        # Unannotated data
        ssl_paths = ['../data/unannotated/train/images']  #will need to be changed according to the data format of the data
        self.ssl_dataset = XRayDataset(ssl_paths, labels=None, augment=True)
        self.ssl_loader = DataLoader(
            self.ssl_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
    



    def train(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(), 
            lr=self.config['ssl']['lr']
        )



        
        for epoch in range(self.config['ssl']['epochs']):
            self.student.train()
            for batch_idx, data in enumerate(self.ssl_loader):
                # Augmentation
                view1 = data.to(self.device)
                view2 = data.to(self.device)  # Set one augmentation
                
                # Student forward
                student_out1 = self.student(view1)
                student_out2 = self.student(view2)
                
                
                # Teacher forward (no gradients) 
                with torch.no_grad():
                    teacher_out1 = self.teacher(view1)
                    teacher_out2 = self.teacher(view2)
                



                # Compute loss
                loss = (self.criterion(student_out1, teacher_out2) + 
                        self.criterion(student_out2, teacher_out1)) / 2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update teacher with EMA
                self.update_teacher()
                
                if batch_idx % 10 == 0:
                    print(f'SSL Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    
    
    def update_teacher(self, momentum=0.996):
        """Update teacher with exponential moving average"""
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data




if __name__ == "__main__":
    # Run with and without FSDP
    print("Training SSL without FSDP...")
    trainer_normal = SSLTrainer(use_fsdp=False)
    trainer_normal.train()
    

    ######## experimental : need to be tested #################
    if torch.cuda.device_count() > 1:
        print("Training SSL with FSDP...")
        trainer_fsdp = SSLTrainer(use_fsdp=True)
        trainer_fsdp.train()
    ######################################################
