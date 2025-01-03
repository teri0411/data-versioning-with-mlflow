import os
import pickle
import numpy as np
import subprocess
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MODEL_PATH, setup_logging

logger = setup_logging()

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
        try:
            # Ensure model directory exists
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            logger.info(f"Created directory: {os.path.dirname(MODEL_PATH)}")
            
            # Save model
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {MODEL_PATH}")
            
            # Add to DVC
            result = subprocess.run(['dvc', 'add', MODEL_PATH], 
                                capture_output=True,
                                text=True,
                                check=True)
            logger.info(f"Model added to DVC tracking: {result.stdout}")
            
            # Git add the .dvc file
            dvc_file = f"{MODEL_PATH}.dvc"
            result = subprocess.run(['git', 'add', dvc_file],
                                capture_output=True,
                                text=True,
                                check=True)
            logger.info(f"Added {dvc_file} to git")
            
            # DVC push
            result = subprocess.run(['dvc', 'push'], 
                                capture_output=True,
                                text=True,
                                check=True)
            logger.info(f"Model pushed to remote storage: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command '{e.cmd}' failed with exit status {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self):
        """Load model from DVC tracked path"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
            
        logger.info(f"Model loaded from {MODEL_PATH}")
        
    def predict(self, X):
        """Make predictions using loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")
            
        return self.model.predict(X)
