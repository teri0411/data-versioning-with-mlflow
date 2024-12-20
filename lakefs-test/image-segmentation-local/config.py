import os

# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# MLflow 설정
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Image Segmentation"

# LakeFS 설정
LAKEFS_ENDPOINT = "http://localhost:8003"
LAKEFS_ACCESS_KEY = "AKIAIOSFOLKFSSAMPLES"
LAKEFS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
LAKEFS_REPO_NAME = "image-segmentation-local-repo"
LAKEFS_BRANCH = "main"

# LakeFS 경로 설정
LAKEFS_MODEL_PATH = "models"
LAKEFS_DATA_PATH = "data"
LAKEFS_IMAGES_PATH = os.path.join(LAKEFS_DATA_PATH, "images")
LAKEFS_MASKS_PATH = os.path.join(LAKEFS_DATA_PATH, "masks")
LAKEFS_MODEL_FILE_PATH = os.path.join(LAKEFS_MODEL_PATH, "model.pth")

# 로컬 디렉토리 설정
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")

# 파일 설정
MODEL_FILENAME = "model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# 학습 설정
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 64
