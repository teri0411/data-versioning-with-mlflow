import dvc.api
import torch
import numpy as np
import pandas as pd
import mlflow
import os
import io

# MLflow 설정
MODEL_NAME = 'Default'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

def load_from_dvc(run_id):
    """MLflow run ID를 기반으로 DVC에서 모델과 데이터를 가져오는 함수"""
    try:
        # MLflow에서 run의 parameters 가져오기
        run = mlflow.get_run(run_id)
        params = run.data.params
        
        # DVC 경로 정보 가져오기
        model_path = params.get('model_path')
        data_path = params.get('data_path')
        repo = params.get('repo')
        version = params.get('version')
        
        print(f"모델 경로: {model_path}")
        print(f"데이터 경로: {data_path}")
        
        # DVC에서 모델 파일 읽기
        model_content = dvc.api.read(
            path=model_path,
            repo=repo,
            rev=version,
            mode='rb'
        )
        
        # 모델 로드
        model_file = io.BytesIO(model_content)
        checkpoint = torch.load(model_file)
        
        # DVC에서 데이터 파일 읽기
        data_content = dvc.api.read(
            path=data_path,
            repo=repo,
            rev=version
        )
        
        # 데이터 로드
        df = pd.read_csv(io.StringIO(data_content))
        
        return checkpoint, df
        
    except Exception as e:
        print(f"DVC에서 파일 로드 중 오류 발생: {str(e)}")
        raise

def predict_wine_quality(input_data, checkpoint):
    """와인 품질 예측 함수"""
    # 스케일러와 모델 가중치 가져오기
    scaler = checkpoint['scaler_state']
    weights = checkpoint['model_state_dict']['linear.weight']
    bias = checkpoint['model_state_dict']['linear.bias']
    
    # 데이터 스케일링
    scaled_data = scaler.transform(input_data)
    
    # PyTorch tensor로 변환 및 예측
    input_tensor = torch.FloatTensor(scaled_data)
    predictions = torch.matmul(input_tensor, weights.t()) + bias
    
    return predictions.numpy()

if __name__ == "__main__":
    try:
        # MLflow run ID 입력 받기 (예: 명령행 인자로)
        run_id = input("MLflow run ID를 입력하세요: ")
        
        # DVC에서 모델과 데이터 로드
        checkpoint, test_data = load_from_dvc(run_id)
        
        # 테스트 데이터 준비
        features = test_data.drop('quality', axis=1)
        sample_data = features.head()  # 처음 5개 샘플만 사용
        
        # 예측 수행
        predictions = predict_wine_quality(sample_data, checkpoint)
        print("\n입력 데이터 shape:", sample_data.shape)
        print("예측 결과:")
        print(predictions)
        
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
