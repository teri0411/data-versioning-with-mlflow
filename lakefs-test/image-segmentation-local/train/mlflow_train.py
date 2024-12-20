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

    def register_model(self, run_id, metrics):
        """모델을 MLflow 모델 레지스트리에 등록합니다."""
        model_name = "image_segmentation"
        
        # 모델 등록
        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name
            )
            print(f"\n=== 모델 등록 완료 ===")
            print(f"모델 이름: {result.name}")
            print(f"버전: {result.version}")
            return result
        except Exception as e:
            print(f"모델 등록 중 오류 발생: {str(e)}")
            return None
