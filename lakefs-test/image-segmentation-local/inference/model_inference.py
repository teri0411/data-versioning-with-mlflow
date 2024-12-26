import os
import torch
import numpy as np
from PIL import Image
from .base_inference import BaseInference
from .mlflow_inference import MLflowInference
from .lakefs_inference import LakeFSInference
from config import *

class ModelInference:
    """Class for managing the entire inference process"""
    
    def __init__(self):
        """Initialize"""
        self.base_inference = BaseInference()
        self.mlflow_inference = MLflowInference()
        self.lakefs_inference = LakeFSInference()
    
    def infer(self, auto_select=True):
        """Perform inference."""
        # Select experiment from MLflow (metadata)
        run = self.mlflow_inference.select_experiment(auto_select)
        if run is None:
            print("Experiment selection cancelled.")
            return
        
        # Get LakeFS model path from MLflow
        model_path = run.data.params.get("model_path")
        if model_path is None:
            raise Exception("Model path not found in MLflow metadata")
        
        # Download model and data from LakeFS
        local_model_path = self.lakefs_inference.download_model(model_path)
        self.lakefs_inference.download_data(run.info.run_id)
        
        # Load model and perform inference
        model = self.base_inference.load_model(local_model_path)
        results = self.base_inference.infer_images()
        
        # Save results
        self.base_inference.save_results(results)
        
        print(f"\nInference complete.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model: {model_path}")
        return results
