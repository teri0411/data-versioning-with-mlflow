import os
import pickle
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MODEL_PATH

class DVCTrain:
    def __init__(self):
        self.model = ElasticNet(random_state=42)
        self.metrics = {}
        
    def preprocess_data(self, data):
        """Preprocess data for training"""
        X = data.drop('quality', axis=1)
        y = data['quality']
        return X, y
        
    def train(self, data):
        """Train model and return metrics"""
        X, y = self.preprocess_data(data)
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        self.metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        return self.metrics
        
    def save_model(self):
        """Save model to DVC tracked path"""
        # Ensure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"Model saved to {MODEL_PATH}")
        
        # Add to DVC
        try:
            os.system(f"dvc add {MODEL_PATH}")
            print(f"Model added to DVC tracking")
        except Exception as e:
            print(f"Warning: Failed to add model to DVC: {e}")
            
    def load_model(self):
        """Load model from DVC tracked path"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
            
        print(f"Model loaded from {MODEL_PATH}")
        
    def predict(self, X):
        """Make predictions using loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")
            
        return self.model.predict(X)
