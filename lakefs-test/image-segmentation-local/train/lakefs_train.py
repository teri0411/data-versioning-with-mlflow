import os
import torch
from config import *
import lakefs_client
from lakefs_client.client import LakeFSClient
from utils.lakefs_utils import upload_to_lakefs

class LakeFSTrain:
    """Class for handling LakeFS-related functionality"""
    
    def __init__(self):
        """Initialize"""
        configuration = lakefs_client.Configuration()
        configuration.host = LAKEFS_ENDPOINT
        configuration.username = LAKEFS_ACCESS_KEY
        configuration.password = LAKEFS_SECRET_KEY
        self.client = LakeFSClient(configuration)
    
    def check_model_exists(self, model_path):
        """Check if model file exists in LakeFS."""
        try:
            # Use SDK instead of lakectl fs stat
            stat = self.client.objects_api.stat_object(
                repository=LAKEFS_REPO_NAME,
                ref=LAKEFS_BRANCH,
                path=model_path
            )
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                return False
            raise e
    
    def save_model(self, model):
        """Save model and upload to LakeFS."""
        print("\nSaving model to LakeFS...")
        torch.save(model.state_dict(), MODEL_PATH)
        
        # Upload model to LakeFS
        lakefs_model_path = os.path.join(LAKEFS_MODEL_PATH, os.path.basename(MODEL_PATH))
        if not self.check_model_exists(lakefs_model_path):
            raise ValueError("Model not found in LakeFS. Please upload the model to LakeFS first.")
        if upload_to_lakefs(self.client, MODEL_PATH, lakefs_model_path):
            return f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{lakefs_model_path}"
        return MODEL_PATH
    
    def upload_data(self):
        """Upload data to LakeFS."""
        print("\nUploading data to LakeFS...")
        
        # Create directories in LakeFS if they don't exist
        lakefs_data_dir = LAKEFS_DATA_PATH
        lakefs_images_dir = LAKEFS_IMAGES_PATH
        lakefs_masks_dir = LAKEFS_MASKS_PATH
        
        # Upload images
        print("- Uploading images...")
        images_dir = os.path.join("data", "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            
        for image_file in os.listdir(images_dir):
            local_path = os.path.join(images_dir, image_file)
            lakefs_path = os.path.join(lakefs_images_dir, image_file)
            upload_to_lakefs(self.client, local_path, lakefs_path)
        
        # Upload masks
        print("- Uploading masks...")
        masks_dir = os.path.join("data", "masks")
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)
            
        for mask_file in os.listdir(masks_dir):
            local_path = os.path.join(masks_dir, mask_file)
            lakefs_path = os.path.join(lakefs_masks_dir, mask_file)
            upload_to_lakefs(self.client, local_path, lakefs_path)
        
        # Upload dataset
        print("- Uploading dataset...")
        dataset_files = ["train.py", "dataset.py", "config.py"]
        for file in dataset_files:
            if os.path.exists(file):
                lakefs_path = os.path.join(lakefs_data_dir, file)
                upload_to_lakefs(self.client, file, lakefs_path)
        
        print("Data upload complete!")
