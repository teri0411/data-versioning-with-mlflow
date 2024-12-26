from config import *
import mlflow

class MLflowTrain:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def log_metrics(self, metrics):
        """Record metrics."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

    def log_params(self, params):
        """Record parameters."""
        mlflow.log_params(params)

    def log_tags(self, tags):
        """Record tags."""
        for key, value in tags.items():
            mlflow.set_tag(key, value)