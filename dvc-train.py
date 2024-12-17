import dvc.api
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get URL from dvc
# wine-quality.csv.dvc file presence is enough if dvc push to remote storage is done
path = "dvc/knuh_v5/data/module/WineQT.csv"
# git init directory location
repo = "https://www.simplatform.com/gitlab2/teri0411/data-versioning.git"
version = "master"  # git tag -a 'v1' -m 'removed 1000 lines' command is required

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)

# PyTorch 모델 클래스 정의
class WineQualityModel(nn.Module):
    def __init__(self, input_dim):
        super(WineQualityModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # read the wine-qualtiy csv file from remote repository
    data = pd.read_csv(data_url, sep=",")
    print(data.columns)  # 컬럼명 확인

    # 데이터를 훈련용과 테스트용으로 나눕니다.
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    # 특성과 타겟 변수로 분리
    train_x = train.drop(["quality"], axis=1)  # quality 컬럼을 제외한 특성
    test_x = test.drop(["quality"], axis=1)  # quality 컬럼을 제외한 특성
    train_y = train[["quality"]]  # quality 컬럼만 타겟
    test_y = test[["quality"]]  # quality 컬럼만 타겟

    # StandardScaler 적용
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # alpha와 l1_ratio 값 처리 (명령어 인자로 전달받기)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # MLflow 실행
    with mlflow.start_run():
        mlflow.log_params({
            "data_path": "dvc/knuh_v5/data/module/WineQT.csv",
            "model_path": "dvc/knuh_v5/data/module/test_model.pt",
            "repo": "https://www.simplatform.com/gitlab2/teri0411/data-versioning.git",
            "version": "master"
        })

        # 데이터 관련 파라미터 로그
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_columns', data.shape[1])

        # ElasticNet 모델 학습
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x_scaled, train_y)

        # ElasticNet 가중치를 PyTorch 모델로 변환
        model = WineQualityModel(input_dim=train_x.shape[1])
        
        # ElasticNet의 계수를 PyTorch 모델의 가중치로 설정
        with torch.no_grad():
            model.linear.weight.copy_(torch.FloatTensor(lr.coef_).reshape(1, -1))
            model.linear.bias.copy_(torch.FloatTensor([lr.intercept_]).reshape(-1))

        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_state': scaler,
            'input_dim': train_x.shape[1]
        }, './test_model.pt')

        # 예측을 위해 데이터를 PyTorch tensor로 변환
        X_test_tensor = torch.FloatTensor(test_x_scaled)
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 예측 수행
        with torch.no_grad():
            predicted_qualities = model(X_test_tensor).numpy()

        # 성능 평가 지표 계산
        rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
        mae = mean_absolute_error(test_y, predicted_qualities)
        r2 = r2_score(test_y, predicted_qualities)

        # 결과 출력
        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # MLflow 로그: 하이퍼파라미터, 성능 지표
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # 모델 저장
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            # 모델 레지스트리에 등록 (파일 저장소가 아닌 경우)
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="terry")
        else:
            mlflow.sklearn.log_model(lr, "model",)
