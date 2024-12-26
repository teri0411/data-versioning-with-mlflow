import mlflow
from config import *

def setup_mlflow():
    """Initialize MLflow settings."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def get_experiment_runs():
    """Get experiment list from MLflow."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return []
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id])

def format_run_info(run):
    """Format experiment information."""
    # Get Git commit hash
    git_commit = run.data.tags.get("mlflow.source.git.commit", "N/A")
    if git_commit != "N/A":
        git_commit = git_commit[:8]  # Display first 8 characters only
    
    # Get parameters and metrics
    params = {k: v for k, v in run.data.params.items()}
    metrics = {k: round(float(v), 4) for k, v in run.data.metrics.items()}
    
    return {
        "run_id": run.info.run_id,
        "git_commit": git_commit,
        "params": params,
        "metrics": metrics
    }

def select_run():
    """Allow user to select an experiment."""
    runs = get_experiment_runs()
    
    if len(runs) == 0:
        print("No saved experiments.")
        return None
    
    print("Available experiments:\n")
    for idx, run_info in runs.iterrows():
        run = mlflow.get_run(run_info.run_id)
        info = format_run_info(run)
        
        print(f"{idx + 1}. Run ID: {info['run_id']}")
        print(f"   Git Commit: {info['git_commit']}")
        print("   Parameters:")
        for k, v in info['params'].items():
            print(f"   - {k}: {v}")
        print("   Metrics:")
        for k, v in info['metrics'].items():
            print(f"   - {k}: {v}")
        print()
    
    # Select experiment
    while True:
        try:
            choice = input("Select experiment number to use: ")
            if not choice.strip():  # Handle empty input
                print("Experiment selection cancelled.")
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(runs):
                run_id = runs.iloc[choice - 1].run_id
                return mlflow.get_run(run_id)
            else:
                print(f"Please enter a number between 1 and {len(runs)}.")
        except ValueError:
            print("Please enter a valid number.")
        except EOFError:
            print("Experiment selection cancelled.")
            return None
