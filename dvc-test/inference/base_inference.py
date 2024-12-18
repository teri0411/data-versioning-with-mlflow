from abc import ABC, abstractmethod
import os
import subprocess
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from model import WineQualityModel

class BaseInference(ABC):
    """Base class for inference"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def ensure_directory(self, file_path: str) -> None:
        """Ensure directory exists for the given file path"""
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def get_file(self, file_path: str) -> None:
        """Get a specific file using DVC"""
        # Ensure directory exists
        self.ensure_directory(file_path)
        
        dvc_file = f"{file_path}.dvc"
        
        # If .dvc file doesn't exist, assume the file is not tracked by DVC
        if not os.path.exists(dvc_file):
            print(f"Warning: {dvc_file} not found, assuming {file_path} is already in place")
            if not os.path.exists(file_path):
                raise Exception(f"Neither {file_path} nor {dvc_file} exists")
            return
        
        try:
            # Try to pull the file using the .dvc file
            subprocess.run(['dvc', 'pull', dvc_file], check=True)
            print(f"Successfully pulled {file_path} using DVC")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to pull {file_path} using DVC: {e}")
            print(f"Will try to use existing {file_path} if available")
            if not os.path.exists(file_path):
                raise Exception(f"Failed to get {file_path} and no local copy exists")
    
    @abstractmethod
    def load_model(self, model_path: str) -> Optional[Tuple[WineQualityModel, object]]:
        """Load model and scaler"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded model"""
        if self.model is None or self.scaler is None:
            raise Exception("Model and scaler must be loaded before prediction")
        
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X_scaled)).numpy()
        
        return predictions
