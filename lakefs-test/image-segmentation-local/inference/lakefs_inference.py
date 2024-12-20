import os
import shutil
from config import *
from utils.lakefs_utils import setup_lakefs_client, download_from_lakefs

class LakeFSInference:
    """LakeFS 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        self.lakefs_client = setup_lakefs_client()
    
    def download_model(self, model_path):
        """LakeFS에서 모델을 다운로드합니다."""
        if model_path.startswith(f"lakefs://{LAKEFS_REPO_NAME}/"):
            # LakeFS 경로에서 실제 경로 추출
            lakefs_path = model_path.replace(f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/", "")
            local_path = MODEL_PATH  # MODEL_PATH는 이미 파일 경로를 포함
            
            # 모델 디렉토리가 없으면 생성
            os.makedirs(MODEL_DIR, exist_ok=True)  # MODEL_DIR만 생성
            
            # LakeFS에서 모델 다운로드
            if download_from_lakefs(self.lakefs_client, lakefs_path, local_path):
                print(f"모델 다운로드 완료: {local_path}")
                return local_path
            else:
                raise Exception("LakeFS에서 모델 다운로드 실패")
                
        return model_path
    
    def download_data(self, run_id):
        """LakeFS에서 데이터를 다운로드합니다."""
        # 이미지와 마스크 디렉토리 생성
        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(MASKS_DIR, exist_ok=True)
        
        # LakeFS에서 데이터 다운로드
        print(f"LakeFS 이미지 경로: {LAKEFS_IMAGES_PATH}")
        print(f"LakeFS 마스크 경로: {LAKEFS_MASKS_PATH}")
        print(f"로컬 이미지 경로: {IMAGES_DIR}")
        print(f"로컬 마스크 경로: {MASKS_DIR}")
        
        if not download_from_lakefs(self.lakefs_client, LAKEFS_IMAGES_PATH, IMAGES_DIR):
            raise Exception("LakeFS에서 이미지 다운로드 실패")
            
        if not download_from_lakefs(self.lakefs_client, LAKEFS_MASKS_PATH, MASKS_DIR):
            raise Exception("LakeFS에서 마스크 다운로드 실패")
            
        print("데이터 다운로드 완료")
