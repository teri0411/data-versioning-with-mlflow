import os
import torch
import subprocess
from config import *
from utils.lakefs_utils import setup_lakefs_client, upload_to_lakefs

class LakeFSTrain:
    """LakeFS 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        self.lakefs_client = setup_lakefs_client()
    
    def check_model_exists(self, model_path):
        """LakeFS에 모델이 존재하는지 확인"""
        try:
            result = subprocess.run(['lakectl', 'fs', 'stat', model_path], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def save_model(self, model):
        """모델을 저장하고 LakeFS에 업로드합니다."""
        print("\nLakeFS에 모델 저장 중...")
        torch.save(model.state_dict(), MODEL_PATH)
        
        # LakeFS에 모델 업로드
        lakefs_model_path = os.path.join(LAKEFS_MODEL_PATH, os.path.basename(MODEL_PATH))
        if not self.check_model_exists(lakefs_model_path):
            raise ValueError("모델이 LakeFS에 없습니다. 먼저 모델을 LakeFS에 업로드해주세요.")
        if upload_to_lakefs(self.lakefs_client, MODEL_PATH, lakefs_model_path):
            return f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{lakefs_model_path}"
        return MODEL_PATH
    
    def upload_data(self):
        """데이터를 LakeFS에 업로드합니다."""
        print("\nLakeFS에 데이터 업로드 중...")
        
        # 이미지 업로드
        print("- 이미지 업로드 중...")
        images_dir = os.path.join("data", "images")
        lakefs_images_dir = os.path.join(LAKEFS_DATA_PATH, "images")
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(".png"):
                continue
            
            local_path = os.path.join(images_dir, image_file)
            lakefs_path = os.path.join(lakefs_images_dir, image_file)
            upload_to_lakefs(self.lakefs_client, local_path, lakefs_path)
        
        # 마스크 업로드
        print("- 마스크 업로드 중...")
        masks_dir = os.path.join("data", "masks")
        lakefs_masks_dir = os.path.join(LAKEFS_DATA_PATH, "masks")
        for mask_file in os.listdir(masks_dir):
            if not mask_file.endswith(".png"):
                continue
            
            local_path = os.path.join(masks_dir, mask_file)
            lakefs_path = os.path.join(lakefs_masks_dir, mask_file)
            upload_to_lakefs(self.lakefs_client, local_path, lakefs_path)
        
        return f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_DATA_PATH}"
