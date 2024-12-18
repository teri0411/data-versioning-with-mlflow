import os
import torch
from typing import Optional, Tuple
from .base_inference import BaseInference
from model import WineQualityModel

class DVCInference(BaseInference):
    def __init__(self):
        super().__init__()
    
    def load_model(self, model_path: str) -> Optional[Tuple[WineQualityModel, object]]:
        """Load model and scaler from checkpoint"""
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Create model instance with input dimension from state dict
            input_dim = checkpoint['model_state_dict']['linear.weight'].shape[1]
            model = WineQualityModel(input_dim=input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model = model
            self.scaler = checkpoint['scaler_state']
            
            return model, checkpoint['scaler_state']
            
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            return None

    def get_model_info(self) -> dict:
        """Get model metadata"""
        return {
            'model_type': 'ElasticNet',
            'source': 'DVC',
            'path': model_path
        }
