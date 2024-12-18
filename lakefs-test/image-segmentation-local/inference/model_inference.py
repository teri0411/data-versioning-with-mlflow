import os
import torch
import numpy as np
from PIL import Image
from .base_inference import BaseInference
from .mlflow_inference import MLflowInference
from .lakefs_inference import LakeFSInference
from config import *

class ModelInference:
    """전체 추론 과정을 관리하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.base_inference = BaseInference()
        self.mlflow_inference = MLflowInference()
        self.lakefs_inference = LakeFSInference()
    
    def infer(self, auto_select=True):
        """추론을 수행합니다."""
        # MLflow에서 실험 선택 (메타데이터)
        run = self.mlflow_inference.select_experiment(auto_select)
        if run is None:
            print("실험 선택이 취소되었습니다.")
            return
        
        # MLflow에서 LakeFS 모델 경로 가져오기
        model_path = run.data.params.get("model_path")
        if not model_path:
            raise Exception("Model path not found in MLflow metadata")
        
        # LakeFS에서 모델과 데이터 다운로드
        local_model_path = self.lakefs_inference.download_model(model_path)
        self.lakefs_inference.download_data(run.info.run_id)
        
        # 모델 로드 및 추론
        model = self.base_inference.load_model(local_model_path)
        predictions = self.base_inference.predict(model)
        
        print(f"\n추론이 완료되었습니다.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model: {model_path}")
        return predictions
