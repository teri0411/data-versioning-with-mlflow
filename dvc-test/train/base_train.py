from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from model import WineQualityModel

class BaseTrain(ABC):
    """Base class for wine quality model training"""
    
    def __init__(self):
        self.model = None  # Will be initialized after data preprocessing
        self.scaler = StandardScaler()
        self.metrics = {}
        
    def preprocess_data(self, data):
        """Preprocess data for training"""
        # Load data if string path is provided
        if isinstance(data, str):
            # Read CSV with proper settings for quoted header
            data = pd.read_csv(data, sep=',', quoting=1)  # QUOTE_ALL for quoted headers
            print("Columns:", data.columns.tolist())
            print("Column types:", data.dtypes)
            
            # Clean column names
            data.columns = [col.strip('"') for col in data.columns]
            print("Cleaned columns:", data.columns.tolist())
            print("Data head:", data.head())
        
        # Split features and target
        X = data.drop('quality', axis=1)  
        y = data['quality']  
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model with input dimension
        if self.model is None:
            self.model = WineQualityModel(input_dim=X.shape[1])
        
        return X_scaled, y
        
    @abstractmethod
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train model and return metrics"""
        pass
        
    @abstractmethod
    def save_model(self) -> None:
        """Save trained model"""
        pass
