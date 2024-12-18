import os
import pandas as pd
from inference.mlflow_inference import MLflowInference

def main():
    """Run inference using model and data from MLflow/DVC"""
    # Initialize MLflow inference
    mlflow_inference = MLflowInference()
    
    # List runs and get user selection
    selected_run = mlflow_inference.list_runs()
    
    # Get paths from the selected run
    DATA_PATH, model_path = mlflow_inference.get_paths()
    
    print(f"\nUsing data: {DATA_PATH}")
    print(f"Using model: {model_path}")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Get files from DVC
    mlflow_inference.get_file(DATA_PATH)
    mlflow_inference.get_file(model_path)
    
    try:
        # Load data
        data = pd.read_csv(DATA_PATH, sep=';')  # Use semicolon separator
        print("Data loaded from:", DATA_PATH)
        print("Columns:", data.columns.tolist())
    except FileNotFoundError:
        print(f"Data file not found at {DATA_PATH}")
        print("Attempting to download data using DVC...")
        mlflow_inference.get_file(DATA_PATH)
        data = pd.read_csv(DATA_PATH, sep=';')  # Use semicolon separator
        print("Data loaded from:", DATA_PATH)
        print("Columns:", data.columns.tolist())
    
    # Load model and make predictions
    mlflow_inference.load_model(model_path)
    X = data.drop('quality', axis=1)
    predictions = mlflow_inference.predict(X)
    
    print("\nPrediction Results:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Sample predictions: {predictions[:5]}")

if __name__ == '__main__':
    main()
