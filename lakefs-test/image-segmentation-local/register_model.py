import argparse
import mlflow
from train.mlflow_train import MLflowTrain
from train.lakefs_train import LakeFSTrain
from config import LAKEFS_REPO_NAME, LAKEFS_BRANCH, MODEL_PATH

def register_model_to_registry(run_id, metrics):
    """Register model to MLflow Model Registry."""
    model_name = "image_segmentation"
    
    # Register model
    try:
        result = mlflow.register_model(
            f"runs:/{run_id}/model",
            model_name
        )
        print(f"\n=== Model Registration Complete ===")
        print(f"Model name: {result.name}")
        print(f"Version: {result.version}")
        return result
    except Exception as e:
        print(f"Error during model registration: {str(e)}")
        return None

class ModelRegistrar:
    """Class responsible for model registration"""
    
    def __init__(self):
        """Initialize"""
        self.mlflow_train = MLflowTrain()
        self.lakefs_train = LakeFSTrain()
    
    def register_model(self, auto_register=True):
        """
        Register model to MLflow Model Registry.

        Args:
            auto_register (bool): Whether to register the model automatically
        """
        # Check if model exists in LakeFS
        if not self.lakefs_train.check_model_exists("models/model.pth"):
            raise Exception("Model not found in LakeFS")
        
        # Check recent experiment results in MLflow
        if auto_register:
            runs = mlflow.search_runs(order_by=["start_time DESC"])
            if len(runs) == 0:
                raise Exception("No MLflow runs found")
            run = runs.iloc[0]
            metrics = {
                "loss": run["metrics.loss"],
                "accuracy": run["metrics.accuracy"]
            }
        else:
            # Manually select experiment
            print("\n=== MLflow Experiment List ===")
            runs = mlflow.search_runs(order_by=["start_time DESC"])
            for i, run in runs.iterrows():
                print(f"{i}. Run ID: {run.run_id}")
                print(f"   Start Time: {run.start_time}")
                print(f"   Loss: {run['metrics.loss']:.4f}")
                print(f"   Accuracy: {run['metrics.accuracy']:.4f}")
                print()
            
            idx = input("Select experiment number to register: ")
            run = runs.iloc[int(idx)]
            metrics = {
                "loss": run["metrics.loss"],
                "accuracy": run["metrics.accuracy"]
            }
        
        # Register model
        result = register_model_to_registry(run.run_id, metrics)
        if result:
            print(f"\nModel has been successfully registered.")
            print(f"Run ID: {run.run_id}")
            print(f"Model Path: {run['params.model_path']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Model Registration')
    parser.add_argument('--manual', action='store_true', help='Manually select the experiment.')
    args = parser.parse_args()
    
    registrar = ModelRegistrar()
    registrar.register_model(auto_register=not args.manual)

if __name__ == "__main__":
    main()
