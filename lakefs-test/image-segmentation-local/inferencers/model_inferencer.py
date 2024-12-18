import os
import torch
import numpy as np
from PIL import Image
from .base_inferencer import BaseInferencer
from .mlflow_inferencer import MLflowInferencer
from .lakefs_inferencer import LakeFSInferencer

class ModelInferencer:
    """전체 추론 과정을 관리하는 클래스"""
    
    def __init__(self):
        self.base_inferencer = BaseInferencer()
        self.mlflow_inferencer = MLflowInferencer()
        self.lakefs_inferencer = LakeFSInferencer()
    
    def infer(self):
        """전체 추론 과정을 실행합니다."""
        # MLflow에서 실험 선택
        run = self.mlflow_inferencer.select_experiment()
        if run is None:
            print("\n실험을 선택하지 않았습니다.")
            return
        
        # 모델과 데이터 경로 가져오기
        model_path = run.data.params.get("model_path")
        data_path = run.data.params.get("data_path")
        
        if not model_path or not data_path:
            print("\n모델 또는 데이터 경로를 찾을 수 없습니다.")
            return
        
        # LakeFS에서 모델과 데이터 다운로드
        print("\nLakeFS에서 모델과 데이터 다운로드 중...")
        model_path = self.lakefs_inferencer.download_model(model_path)
        data_path = self.lakefs_inferencer.download_data(data_path, run.info.run_id)
        
        # 모델 가중치 로드
        print("\n모델 가중치 로드 중...")
        self.base_inferencer.load_model_weights(model_path)
        
        # 추론 실행
        print("\n추론 시작...")
        images_dir = os.path.join(data_path, "images")
        masks_dir = os.path.join(data_path, "masks")
        
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(".png"):
                continue
            
            image_path = os.path.join(images_dir, image_file)
            mask_path = os.path.join(masks_dir, image_file.replace("image", "mask"))
            
            # 추론 수행
            output = self.base_inferencer.infer_image(image_path)
            output = torch.sigmoid(output).numpy()
            output = (output > 0.5).astype(np.uint8) * 255
            
            # 결과 저장
            output_image = Image.fromarray(output)
            output_dir = os.path.join(data_path, "predictions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, image_file.replace("image", "pred"))
            output_image.save(output_path)
            
            print(f"이미지 처리 완료: {image_file}")
        
        print("\n추론이 완료되었습니다.")
