import os
import shutil
from config import *
from utils.lakefs_utils import setup_lakefs_client, download_from_lakefs

class LakeFSInference:
    """Class for handling LakeFS-related functionality"""
    
    def __init__(self):
        self.lakefs_client = setup_lakefs_client()
    
    def download_model(self, model_path):
        """Download model from LakeFS."""
        if model_path.startswith(f"lakefs://{LAKEFS_REPO_NAME}/"):
            # Extract actual path from LakeFS path
            lakefs_path = model_path.replace(f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/", "")
            local_path = MODEL_PATH  # MODEL_PATH already includes file path
            
            # Create model directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)  # Only create MODEL_DIR
            
            # Download model from LakeFS
            if download_from_lakefs(self.lakefs_client, lakefs_path, local_path):
                print(f"Model download complete: {local_path}")
                return local_path
            else:
                raise Exception("Failed to download model from LakeFS")
                
        return model_path
    
    def download_data(self, run_id):
        """Download data from LakeFS."""
        # Create image and mask directories
        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(MASKS_DIR, exist_ok=True)
        
        # Download data from LakeFS
        print(f"LakeFS images path: {LAKEFS_IMAGES_PATH}")
        print(f"LakeFS masks path: {LAKEFS_MASKS_PATH}")
        print(f"Local images path: {IMAGES_DIR}")
        print(f"Local masks path: {MASKS_DIR}")
        
        if not download_from_lakefs(self.lakefs_client, LAKEFS_IMAGES_PATH, IMAGES_DIR):
            raise Exception("Failed to download images from LakeFS")
            
        if not download_from_lakefs(self.lakefs_client, LAKEFS_MASKS_PATH, MASKS_DIR):
            raise Exception("Failed to download masks from LakeFS")
            
        print("Data download complete")
