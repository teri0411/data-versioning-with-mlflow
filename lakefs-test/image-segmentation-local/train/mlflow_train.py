import mlflow
from config import *
from utils.git_utils import get_git_commit_hash

class MLflowTrain:
    """MLflow 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        """MLflow 설정을 초기화합니다."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def log_params(self, params=None):
        """학습 파라미터와 Git 정보를 기록합니다."""
        # Git commit hash 기록
        git_commit_hash = get_git_commit_hash()
        if git_commit_hash:
            mlflow.set_tag("git_commit_hash", git_commit_hash)
        
        # 사용자 지정 파라미터가 있으면 기록
        if params:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """메트릭을 기록합니다."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
