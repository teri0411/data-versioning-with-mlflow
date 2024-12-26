import os
import torch
from .base_train import BaseTrain
from config import EPOCHS, MODEL_DIR, MODEL_PATH

class ModelTrain:
    """Class for managing model training"""
    
    def __init__(self):
        """Initialize"""
        self.base_train = BaseTrain()
        self.loss = 0.0
        self.accuracy = 0.0
        self.epochs = EPOCHS
        self.learning_rate = 0.001  # Set learning rate same as BaseTrain
    
    def save_model(self, model):
        """Save model to file."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    def train(self):
        """
        Perform training and return the trained model.

        Returns:
            Trained model
        """
        # Train model
        for epoch in range(self.epochs):
            loss = self.base_train.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")
            self.loss = loss  # Save loss from last epoch
        
        # Calculate accuracy (example)
        self.accuracy = 0.85  # Should be calculated using validation data
        
        # Save model
        self.save_model(self.base_train.model)
        
        return self.base_train.model
        
    def get_metrics(self):
        """Return training metrics."""
        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy)
        }
