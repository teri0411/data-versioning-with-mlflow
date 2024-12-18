import os
import pandas as pd
import mlflow
import requests
import subprocess
from train.dvc_train import DVCTrain
from config import DATA_PATH, MODEL_PATH, MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from utils import get_git_commit_info, get_dvc_paths

def ensure_data():
    """Ensure data exists by checking DVC or downloading"""
    dvc_file = f"{DATA_PATH}.dvc"
    
    # 1. Check if data is tracked by DVC
    if os.path.exists(dvc_file):
        print(f"Data is tracked by DVC. Attempting to pull from {DATA_PATH}")
        try:
            subprocess.run(['dvc', 'pull', dvc_file], check=True)
            print(f"Successfully pulled data from DVC")
            return
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to pull data using DVC: {e}")
            print("Will try to use existing file or download")
    
    # 2. Check if data file exists locally
    if os.path.exists(DATA_PATH):
        print(f"Using existing data file: {DATA_PATH}")
        return
    
    # 3. Download data if not available
    print(f"Data not found. Downloading from UCI...")
    download_wine_data()

def download_wine_data():
    """Download wine quality dataset from UCI"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"Downloading wine quality dataset from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        # Save data
        with open(DATA_PATH, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully downloaded data to {DATA_PATH}")
            
    except Exception as e:
        raise Exception(f"Failed to download data: {e}")

def main():
    """Train model and log to MLflow with DVC paths"""
    # Ensure data exists
    ensure_data()
    
    # Load data
    data = pd.read_csv(DATA_PATH, sep=';')  # Use semicolon separator
    print("Data loaded from:", DATA_PATH)
    print("Columns:", data.columns.tolist())
    
    # Train model
    trainer = DVCTrain()
    metrics = trainer.train(data)
    trainer.save_model()
    print(f"Training completed. Metrics: {metrics}")
    
    # Get DVC paths and Git info
    dvc_paths = get_dvc_paths()
    git_commit = get_git_commit_info()
    
    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log DVC paths
        mlflow.log_params({
            'data_path': dvc_paths['data_path'],
            'model_path': dvc_paths['model_path']
        })
        
        # Log Git commit
        if git_commit:
            mlflow.set_tag('git_commit', git_commit)
        
        # Log model parameters
        mlflow.log_params({
            'model_type': 'ElasticNet',
            'input_features': data.drop('quality', axis=1).columns.tolist()
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        print("Training info logged to MLflow")

if __name__ == '__main__':
    main()
