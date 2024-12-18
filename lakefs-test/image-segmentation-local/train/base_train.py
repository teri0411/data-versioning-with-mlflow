import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from model import create_model
from dataset import get_data_loader

class BaseTrain:
    """기본 모델 학습 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.dataloader = get_data_loader()
    
    def train_epoch(self, epoch):
        """한 에폭 동안의 학습을 수행합니다."""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for images, masks in self.dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        return avg_loss
