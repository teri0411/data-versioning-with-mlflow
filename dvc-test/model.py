import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class WineQualityModel(nn.Module):
    """와인 품질 예측을 위한 PyTorch 모델"""
    def __init__(self, input_dim: int):
        super(WineQualityModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> Dict[str, float]:
        """Train the model and return metrics"""
        # Convert data to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        
        # Train
        for epoch in range(epochs):
            # Forward pass
            y_pred = self(X_tensor)
            loss = criterion(y_pred, y_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            y_pred = self(X_tensor).numpy()
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return metrics

class WineQualityPredictor:
    """와인 품질 예측을 위한 클래스"""
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        
    def predict(self, input_data):
        """
        와인 품질 예측 함수
        
        Args:
            input_data: 예측할 데이터
            
        Returns:
            예측된 와인 품질 점수
        """
        # 스케일러와 모델 가중치 가져오기
        scaler = self.checkpoint['scaler_state']
        weights = self.checkpoint['model_state_dict']['linear.weight']
        bias = self.checkpoint['model_state_dict']['linear.bias']
        
        # 데이터 스케일링
        scaled_data = scaler.transform(input_data)
        
        # PyTorch tensor로 변환 및 예측
        input_tensor = torch.FloatTensor(scaled_data)
        predictions = torch.matmul(input_tensor, weights.t()) + bias
        
        return predictions.numpy()

def train_elasticnet(train_x, train_y, alpha=0.5, l1_ratio=0.5):
    """ElasticNet 모델 학습"""
    # 데이터 스케일링
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    
    # ElasticNet 모델 학습
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x_scaled, train_y)
    
    return lr, scaler

def convert_to_pytorch(lr_model, input_dim):
    """ElasticNet 모델을 PyTorch 모델로 변환"""
    model = WineQualityModel(input_dim=input_dim)
    with torch.no_grad():
        model.linear.weight.copy_(torch.FloatTensor(lr_model.coef_).reshape(1, -1))
        model.linear.bias.copy_(torch.FloatTensor([lr_model.intercept_]).reshape(-1))
    return model
