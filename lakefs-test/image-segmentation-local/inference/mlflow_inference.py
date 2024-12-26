import mlflow
from config import *

class MLflowInference:
    """Class for handling MLflow-related functionality"""
    
    def __init__(self):
        """Initialize"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def select_experiment(self, auto_select=True):
        """Select an experiment."""
        print("\nMLflow Experiment List:")
        runs = mlflow.search_runs(order_by=["start_time DESC"])
        
        if len(runs) == 0:
            print("No experiments found.")
            return None
            
        # Print experiment list
        for idx, (_, run) in enumerate(runs.iterrows()):
            run_details = mlflow.get_run(run.run_id)
            print(f"\n{idx + 1}. Run ID: {run.run_id}")
            print(f"   Git Commit: {run_details.data.tags.get('mlflow.source.git.commit', 'N/A')}")
            print(f"   Start Time: {run.start_time}")
            
            # Parameters
            params = run_details.data.params
            if params:
                print("   Parameters:")
                for param_name, value in params.items():
                    print(f"   - {param_name}: {value}")
            
            # Metrics
            metrics = run_details.data.metrics
            if metrics:
                print("   Metrics:")
                for metric_name, value in metrics.items():
                    print(f"   - {metric_name}: {value:.4f}")
        
        # Select experiment
        if auto_select:
            selected_run = mlflow.get_run(runs.iloc[0].run_id)
            print(f"\nSelected most recent experiment:")
            print(f"- Run ID: {selected_run.info.run_id}")
            print(f"- Git Commit: {selected_run.data.tags.get('mlflow.source.git.commit', 'N/A')}")
            print(f"- Start Time: {selected_run.info.start_time}")
            return selected_run
            
        while True:
            try:
                choice = input("\nSelect experiment number (default: 1): ").strip()
                if not choice:
                    choice = "1"
                choice = int(choice)
                if 1 <= choice <= len(runs):
                    selected_run = mlflow.get_run(runs.iloc[choice-1].run_id)
                    print(f"\nSelected experiment:")
                    print(f"- Run ID: {selected_run.info.run_id}")
                    print(f"- Git Commit: {selected_run.data.tags.get('mlflow.source.git.commit', 'N/A')}")
                    print(f"- Start Time: {selected_run.info.start_time}")
                    return selected_run
                else:
                    print("Please select a valid number.")
            except ValueError:
                print("Please enter a number.")
