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
    
    def log_params(self, params=None):
        """학습 파라미터와 Git 정보를 기록합니다."""
        # Git commit hash 기록
        git_commit_hash = get_git_commit_hash()
        if git_commit_hash:
            mlflow.set_tag("git_commit_hash", git_commit_hash)
        
        # 기본 학습 파라미터 기록
        default_params = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model_path": f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth"
        }
        
        # 사용자 지정 파라미터가 있으면 업데이트
        if params:
            default_params.update(params)
        
        mlflow.log_params(default_params)
    
    def log_metrics(self, metrics):
        """메트릭을 기록합니다."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
    
    def end_run(self):
        """MLflow 실험을 종료합니다."""
        mlflow.end_run()
    
    def register_model(self, run_id, metrics):
        """모델을 MLflow Model Registry에 등록합니다."""
        model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth"
        print(f"\n모델 등록 중...")
        print(f"Run ID: {run_id}")
        print(f"Metrics: {metrics}")
        print(f"Model Path: {model_path}")
        
        try:
            # MLflow에 모델 메타데이터 등록
            client = mlflow.MlflowClient()
            
            # 등록된 모델이 없으면 생성
            try:
                client.create_registered_model("image_segmentation_model")
            except Exception:
                pass  # 이미 존재하는 경우 무시
            
            # 새로운 버전 생성
            version = client.create_model_version(
                name="image_segmentation_model",
                source=model_path,  # LakeFS 경로
                run_id=run_id,
                tags={
                    "storage": "lakefs",
                    "accuracy": str(metrics.get("accuracy", 0)),
                    "loss": str(metrics.get("loss", 0))
                }
            )
            print(f"모델이 성공적으로 등록되었습니다. (버전: {version.version})")
        except Exception as e:
            print(f"모델 등록 실패: {str(e)}")
            raise
