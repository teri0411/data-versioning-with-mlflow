import os

# MLflow 설정
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Image Segmentation"

# LakeFS 설정
LAKEFS_ENDPOINT = "http://localhost:8003"
LAKEFS_ACCESS_KEY = "AKIAIOSFOLKFSSAMPLES"
LAKEFS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
LAKEFS_REPO_NAME = "image-segmentation-local-repo"
LAKEFS_BRANCH = "main"

# 모델 설정
MODEL_DIR = "models"
MODEL_FILENAME = "model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# 데이터 설정
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")

# 학습 설정
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 64
