import os
import mlflow
import torch
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from .base_inference import BaseInference
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from model import WineQualityModel
import dvc.api

class MLflowInference(BaseInference):
    def __init__(self):
        super().__init__()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = None
        self.scaler = None
        self.selected_run = None
    
    def list_runs(self) -> mlflow.entities.Run:
        """List all MLflow runs and let user select one"""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            raise Exception(f"Experiment {EXPERIMENT_NAME} not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            raise Exception("No runs found in the experiment")
        
        print("\nAvailable MLflow runs:")
        print("-" * 80)
        for i, run in enumerate(runs):
            start_time = run.info.start_time / 1000  # Convert to seconds
            metrics = run.data.metrics
            params = run.data.params
            git_commit = run.data.tags.get('git_commit', 'N/A')
            print(f"\n{i+1}. Run ID: {run.info.run_id}")
            print(f"   Started: {pd.Timestamp(start_time, unit='s')}")
            print(f"   Git Commit: {git_commit}")
            print(f"   Metrics: {metrics}")
            print(f"   Model Path: {params.get('model_path', 'N/A')}")
            print(f"   Data Path: {params.get('data_path', 'N/A')}")
        print("-" * 80)
        
        # 자동으로 가장 최근 실행을 선택
        self.selected_run = runs[0]
        print(f"\nAutomatically selected most recent run (ID: {self.selected_run.info.run_id})")
        return self.selected_run
    
    def get_paths(self) -> Tuple[str, str]:
        """Get data and model paths from selected run"""
        if not self.selected_run:
            raise Exception("No run selected. Call list_runs() first")
            
        data_path = self.selected_run.data.params.get('data_path')
        model_path = self.selected_run.data.params.get('model_path')
        
        if not data_path or not model_path:
            raise Exception("Selected run is missing data_path or model_path parameters")
            
        return data_path, model_path
    
    def load_model(self, model_path: str) -> Optional[Tuple[WineQualityModel, object]]:
        """Load model and scaler from checkpoint"""
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Create model instance with input dimension from state dict
            input_dim = checkpoint['model_state_dict']['linear.weight'].shape[1]
            model = WineQualityModel(input_dim=input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model = model
            self.scaler = checkpoint['scaler_state']
            
            return model, checkpoint['scaler_state']
            
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            return None
    
    def load_data(self, data_path: str) -> str:
        """Load data using DVC"""
        try:
            with dvc.api.open(data_path) as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load from DVC: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded model"""
        if self.model is None or self.scaler is None:
            raise Exception("Model and scaler must be loaded before prediction")
        
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X_scaled)).numpy()
        
        return predictions
