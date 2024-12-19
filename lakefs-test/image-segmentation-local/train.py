from utils.dir_utils import ensure_directories
from train.model_train import ModelTrain
from train.mlflow_train import MLflowTrain
from config import LAKEFS_REPO_NAME, LAKEFS_BRANCH, MODEL_FILENAME
from utils.lakefs_utils import setup_lakefs_client, upload_to_lakefs
import mlflow

def main():
    """메인 함수"""
    # 디렉토리 생성
    ensure_directories()
    
    # LakeFS 클라이언트 설정
    client = setup_lakefs_client(create_if_not_exists=True)
    
    # MLflow 설정
    
    # 모델 학습
    trainer = ModelTrain()
    model = trainer.train()  # model.pth 생성

    # LakeFS에 모델과 데이터 업로드
    upload_to_lakefs(client, "models/model.pth", "models/model.pth")
    upload_to_lakefs(client, "data/images", "data/images")
    upload_to_lakefs(client, "data/masks", "data/masks")
    
    print("\n=== 학습 완료 ===")
    print(f"LakeFS에 모델이 등록되었습니다. 파일명:{MODEL_FILENAME}")
    

    mlflow_train = MLflowTrain()
    # MLflow 실험 시작
    with mlflow.start_run():
        # LakeFS 경로
        lakefs_model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth"
        lakefs_images_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/data/images"
        lakefs_masks_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/data/masks"
        
        print(f"모델 경로: {lakefs_model_path}")
        print(f"이미지 경로: {lakefs_images_path}")
        print(f"마스크 경로: {lakefs_masks_path}")
        
        # MLflow에 실험 기록
        metrics = {"loss": trainer.loss, "accuracy": trainer.accuracy}
        parameters = {
            "model_path": lakefs_model_path,
            "images_path": lakefs_images_path,
            "masks_path": lakefs_masks_path,
            "epochs": trainer.epochs,
            "learning_rate": trainer.learning_rate
        }
        mlflow_train.log_metrics(metrics)
        mlflow_train.log_params(parameters)
    
    return model

if __name__ == "__main__":
    main()
