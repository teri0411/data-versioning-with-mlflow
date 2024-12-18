import os
import torch
from .base_train import BaseTrain
from config import MODEL_PATH

class DVCTrain(BaseTrain):
    """DVC-based training for wine quality prediction"""
    
    def __init__(self):
        super().__init__()
        
    def train(self, data):
        """Train model and track with DVC"""
        # Preprocess data
        X_scaled, y = self.preprocess_data(data)
        
        # Train model
        self.metrics = self.model.fit(X_scaled, y)
        
        return self.metrics
        
    def save_model(self):
        """Save model checkpoint"""
        # Create checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler,
            'metrics': self.metrics
        }
        
        # Save checkpoint
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(checkpoint, MODEL_PATH)
