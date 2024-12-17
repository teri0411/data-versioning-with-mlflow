import dvc.api
import torch
import numpy as np
import pandas as pd
import mlflow
import os
import io
from typing import Tuple, Dict, Any

# MLflow 설정
MODEL_NAME = 'Default'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

class WineQualityPredictor:
    """와인 품질 예측을 위한 클래스"""
    
    def __init__(self, run_id: str):
        """
        Args:
            run_id: MLflow run ID
        """
        self.run_id = run_id
        self.checkpoint, self.data = self._load_from_dvc()
    
    def _load_from_dvc(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """MLflow run ID를 기반으로 DVC에서 모델과 데이터를 가져오는 함수"""
        try:
            # MLflow에서 run의 parameters 가져오기
            run = mlflow.get_run(self.run_id)
            params = run.data.params
            
            # DVC 경로 정보 가져오기
            paths = {
                'model': params.get('model_path'),
                'data': params.get('data_path'),
                'repo': params.get('repo'),
                'version': params.get('version')
            }
            
            print(f"모델 경로: {paths['model']}")
            print(f"데이터 경로: {paths['data']}")
            
            # 모델 로드
            model_content = dvc.api.read(
                path=paths['model'],
                repo=paths['repo'],
                rev=paths['version'],
                mode='rb'
            )
            checkpoint = torch.load(io.BytesIO(model_content))
            
            # 데이터 로드
            data_content = dvc.api.read(
                path=paths['data'],
                repo=paths['repo'],
                rev=paths['version']
            )
            df = pd.read_csv(io.StringIO(data_content))
            
            return checkpoint, df
            
        except Exception as e:
            print(f"DVC에서 파일 로드 중 오류 발생: {str(e)}")
            raise
    
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
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

def main():
    try:
        # MLflow run ID 입력 받기
        run_id = input("MLflow run ID를 입력하세요: ")
        
        # 예측기 초기화
        predictor = WineQualityPredictor(run_id)
        
        # 테스트 데이터 준비
        features = predictor.data.drop('quality', axis=1)
        sample_data = features.head()  # 처음 5개 샘플만 사용
        
        # 예측 수행
        predictions = predictor.predict(sample_data)
        print("\n입력 데이터 shape:", sample_data.shape)
        print("예측 결과:")
        print(predictions)
        
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
