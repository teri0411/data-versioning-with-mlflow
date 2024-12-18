import os
from config import *
from utils.lakefs_utils import setup_lakefs_client, download_from_lakefs

class LakeFSInferencer:
    """LakeFS 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        self.lakefs_client = setup_lakefs_client()
    
    def download_model(self, model_path):
        """LakeFS에서 모델을 다운로드합니다."""
        if model_path.startswith(f"lakefs://{LAKEFS_REPO_NAME}/"):
            # LakeFS 경로에서 실제 경로 추출
            lakefs_path = model_path.replace(f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/", "")
            if not download_from_lakefs(self.lakefs_client, lakefs_path, MODEL_PATH):
                raise Exception("Failed to download model from LakeFS")
            return MODEL_PATH
        return model_path
    
    def download_data(self, data_path, run_id):
        """LakeFS에서 데이터를 다운로드합니다."""
        if not data_path.startswith(f"lakefs://{LAKEFS_REPO_NAME}/"):
            return data_path
        
        # 로컬 디렉토리 생성
        local_data_dir = os.path.join("data", run_id)
        local_images_dir = os.path.join(local_data_dir, "images")
        local_masks_dir = os.path.join(local_data_dir, "masks")
        os.makedirs(local_images_dir, exist_ok=True)
        os.makedirs(local_masks_dir, exist_ok=True)
        
        # LakeFS에서 데이터 다운로드
        lakefs_path = data_path.replace(f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/", "")
        lakefs_images_dir = os.path.join(lakefs_path, "images")
        lakefs_masks_dir = os.path.join(lakefs_path, "masks")
        
        print(f"LakeFS에서 이미지 다운로드 중... ({lakefs_images_dir})")
        for image_file in ["image_0.png", "image_1.png", "image_2.png", "image_3.png", "image_4.png"]:
            lakefs_path = os.path.join(lakefs_images_dir, image_file)
            local_path = os.path.join(local_images_dir, image_file)
            download_from_lakefs(self.lakefs_client, lakefs_path, local_path)
        
        print(f"LakeFS에서 마스크 다운로드 중... ({lakefs_masks_dir})")
        for mask_file in ["mask_0.png", "mask_1.png", "mask_2.png", "mask_3.png", "mask_4.png"]:
            lakefs_path = os.path.join(lakefs_masks_dir, mask_file)
            local_path = os.path.join(local_masks_dir, mask_file)
            download_from_lakefs(self.lakefs_client, lakefs_path, local_path)
        
        return local_data_dir
