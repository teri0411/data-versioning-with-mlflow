import os
from config import *

def ensure_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 생성합니다."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
