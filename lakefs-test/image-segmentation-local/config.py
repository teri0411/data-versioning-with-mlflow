import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Image Segmentation"

# LakeFS configuration
LAKEFS_ENDPOINT = "http://localhost:8003"
LAKEFS_ACCESS_KEY = "AKIAIOSFOLKFSSAMPLES" # this is a sample access key (https://github.com/treeverse/lakeFS-samples)
LAKEFS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # this is a sample secret key (https://github.com/treeverse/lakeFS-samples)
LAKEFS_REPO_NAME = "image-segmentation-local-repo"
LAKEFS_BRANCH = "main"

# LakeFS path configuration
LAKEFS_MODEL_PATH = "models"
LAKEFS_DATA_PATH = "data"
LAKEFS_IMAGES_PATH = os.path.join(LAKEFS_DATA_PATH, "images")
LAKEFS_MASKS_PATH = os.path.join(LAKEFS_DATA_PATH, "masks")
LAKEFS_MODEL_FILE_PATH = os.path.join(LAKEFS_MODEL_PATH, "model.pth")

# Local directory configuration
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")

# File configuration
MODEL_FILENAME = "model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Training configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 64
