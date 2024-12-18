import mlflow
from config import *
from utils.git_utils import get_git_commit_hash

class MLflowTrain:
    """MLflow 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def start_run(self):
        """MLflow 실험을 시작합니다."""
        return mlflow.start_run()
    
    def log_params(self):
        # Git commit hash 기록
        git_commit_hash = get_git_commit_hash()
        if git_commit_hash:
            mlflow.set_tag("git_commit_hash", git_commit_hash)
        
        # 학습 파라미터 기록
        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        })
    
    def log_metrics(self, metrics):
        """메트릭을 기록합니다."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
    
    def log_model_path(self, model_path):
        """모델 경로를 기록합니다."""
        mlflow.log_param("model_path", model_path)
    
    def log_data_path(self):
        """데이터 경로를 기록합니다."""
        data_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_DATA_PATH}"
        mlflow.log_param("data_path", data_path)
