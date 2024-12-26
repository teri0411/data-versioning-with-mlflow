import os
from config import *

def ensure_directories():
    """Check if required directories exist and create them if they don't."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
