import argparse
import mlflow
from train.mlflow_train import MLflowTrain
from train.dvc_train import DVCTrain
from config import *
from utils import get_dvc_paths

class ModelRegistrar:
    """Class responsible for model registration"""
    
    def __init__(self):
        """Initialize"""
        self.mlflow_train = MLflowTrain()
        self.dvc_train = DVCTrain()
    
    def register_model(self, auto_register=True):
        """
        Register model to MLflow Model Registry.

        Args:
            auto_register (bool): Whether to register the model automatically
        """
        # Check if model exists in DVC
        if not os.path.exists(MODEL_PATH):
            raise Exception("Model not found in local path")
            
        dvc_file = f"{MODEL_PATH}.dvc"
        if not os.path.exists(dvc_file):
            raise Exception("Model is not tracked by DVC")
        
        # Check recent experiment results in MLflow
        if auto_register:
            runs = mlflow.search_runs(order_by=["start_time DESC"])
            if len(runs) == 0:
                raise Exception("No MLflow runs found")
            run = runs.iloc[0]
        else:
            # Manually select experiment
            print("\n=== MLflow Experiment List ===")
            runs = mlflow.search_runs(order_by=["start_time DESC"])
            for i, run in runs.iterrows():
                print(f"{i}. Run ID: {run.run_id}")
                print(f"   Start Time: {run.start_time}")
                print(f"   Metrics: {run.metrics}")
                print()
            
            idx = input("Select experiment number to register: ")
            run = runs.iloc[int(idx)]
        
        # Get DVC path
        dvc_paths = get_dvc_paths()
        
        # Register model metadata
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_id=run.run_id):
            mlflow.register_model(
                model_uri=f"runs:/{run.run_id}/model",
                name="wine_quality_model",
                tags={"source_run": run.run_id}
            )
        
        print(f"\nModel has been successfully registered.")
        print(f"Run ID: {run.run_id}")
        print(f"Model Path: {dvc_paths['model_path']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Model Registration')
    parser.add_argument('--manual', action='store_true', help='Manually select the experiment.')
    args = parser.parse_args()
    
    registrar = ModelRegistrar()
    registrar.register_model(auto_register=not args.manual)

if __name__ == "__main__":
    main()
