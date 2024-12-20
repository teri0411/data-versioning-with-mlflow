from config import *
import mlflow

class MLflowTrain:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def log_metrics(self, metrics):
        """메트릭을 기록합니다."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

    def log_params(self, params):
        """파라미터를 기록합니다."""
        mlflow.log_params(params)

    def log_tags(self, tags):
        """태그를 기록합니다."""
        for key, value in tags.items():
            mlflow.set_tag(key, value)