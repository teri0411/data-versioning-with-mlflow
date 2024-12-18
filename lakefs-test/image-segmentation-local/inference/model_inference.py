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
        # MLflow에서 실험 선택
        run = self.mlflow_inference.select_experiment(auto_select)
        if run is None:
            print("실험 선택이 취소되었습니다.")
            return
            
        # 모델과 데이터 다운로드
        model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth"
        model_path = self.lakefs_inference.download_model(model_path)
        self.lakefs_inference.download_data(run.info.run_id)
        
        # 모델 로드
        self.base_inference.load_model(model_path)
        
        # 추론 수행
        results = self.base_inference.infer_images()
        
        # 결과 저장
        self.base_inference.save_results(results)
