import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.sklearn
import dvc.api
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# 로깅 설정
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# 상수 정의
DVC_CONFIG = {
    "path": "dvc/knuh_v5/data/module/WineQT.csv",
    "repo": "https://www.simplatform.com/gitlab2/teri0411/data-versioning.git",
    "rev": "master"
}

MLFLOW_CONFIG = {
    "model_name": "Default",
    "model_save_path": "dvc/knuh_v5/data/module/test_model.pt"
}

class WineQualityModel(nn.Module):
    """와인 품질 예측을 위한 PyTorch 모델"""
    def __init__(self, input_dim):
        super(WineQualityModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def load_data():
    """DVC에서 데이터를 로드하고 전처리하는 함수"""
    data_url = dvc.api.get_url(**DVC_CONFIG)
    data = pd.read_csv(data_url, sep=",")
    
    # 데이터 분할
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    
    # 특성과 타겟 분리
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    return train_x, test_x, train_y, test_y

def train_model(train_x, train_y, alpha=0.5, l1_ratio=0.5):
    """ElasticNet 모델 학습 및 PyTorch 모델로 변환"""
    # 데이터 스케일링
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    
    # ElasticNet 모델 학습
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x_scaled, train_y)
    
    # PyTorch 모델로 변환
    model = WineQualityModel(input_dim=train_x.shape[1])
    with torch.no_grad():
        model.linear.weight.copy_(torch.FloatTensor(lr.coef_).reshape(1, -1))
        model.linear.bias.copy_(torch.FloatTensor([lr.intercept_]).reshape(-1))
    
    return model, scaler, lr

def evaluate_model(model, test_x, test_y, scaler):
    """모델 성능 평가"""
    test_x_scaled = scaler.transform(test_x)
    X_test_tensor = torch.FloatTensor(test_x_scaled)
    
    model.eval()
    with torch.no_grad():
        predicted_qualities = model(X_test_tensor).numpy()
    
    rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
    mae = mean_absolute_error(test_y, predicted_qualities)
    r2 = r2_score(test_y, predicted_qualities)
    
    return rmse, mae, r2

def save_model(model, scaler, train_x):
    """모델 저장"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler,
        'input_dim': train_x.shape[1]
    }, MLFLOW_CONFIG['model_save_path'])

def main():
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # MLflow 설정
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    # 하이퍼파라미터 설정
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # 데이터 로드
    train_x, test_x, train_y, test_y = load_data()
    
    # MLflow 실행
    with mlflow.start_run():
        # 모델 학습
        model, scaler, lr = train_model(train_x, train_y, alpha, l1_ratio)
        
        # 모델 평가
        rmse, mae, r2 = evaluate_model(model, test_x, test_y, scaler)
        
        # 모델 저장
        save_model(model, scaler, train_x)
        
        # MLflow 로깅
        mlflow.log_params({
            "data_path": DVC_CONFIG["path"],
            "model_path": MLFLOW_CONFIG["model_save_path"],
            "repo": DVC_CONFIG["repo"],
            "version": DVC_CONFIG["rev"],
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "input_rows": len(train_x) + len(test_x),
            "input_columns": train_x.shape[1]
        })
        
        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })
        
        # 모델 저장 (MLflow)
        if urlparse(mlflow.get_tracking_uri()).scheme != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="terry")
        else:
            mlflow.sklearn.log_model(lr, "model")
        
        # 결과 출력
        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

if __name__ == "__main__":
    main()
