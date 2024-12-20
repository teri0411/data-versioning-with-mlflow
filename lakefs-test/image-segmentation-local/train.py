from utils.dir_utils import ensure_directories
from train.model_train import ModelTrain
from train.mlflow_train import MLflowTrain
from config import LAKEFS_REPO_NAME, LAKEFS_BRANCH, MODEL_FILENAME
from utils.lakefs_utils import setup_lakefs_client, upload_to_lakefs, commit_to_lakefs_with_git
import mlflow
import os

def main():
    # 필요한 디렉토리 생성
    ensure_directories()
    
    # LakeFS 클라이언트 설정
    client = setup_lakefs_client(create_if_not_exists=True)
    
    # 모델 학습
    model_train = ModelTrain()
    model = model_train.train()  # 모델 반환
    metrics = model_train.get_metrics()  # 메트릭 가져오기
    
    # 절대 경로 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/model.pth")
    images_path = os.path.join(current_dir, "data/images")
    masks_path = os.path.join(current_dir, "data/masks")

    # LakeFS에 모델과 데이터 업로드 (하나의 커밋으로)
    uploaded_files = []
    changes_detected = False
    
    # 모델 파일 업로드 (변경된 경우에만)
    if upload_to_lakefs(client, model_path, "models/model.pth"):
        uploaded_files.append("models/model.pth")
        changes_detected = True
    
    # 데이터 파일 업로드 (변경된 경우에만)
    if upload_to_lakefs(client, images_path, "data/images"):
        uploaded_files.append("data/images/*")
        changes_detected = True
    
    if upload_to_lakefs(client, masks_path, "data/masks"):
        uploaded_files.append("data/masks/*")
        changes_detected = True
    
    if changes_detected:
        # Git 정보를 포함하여 LakeFS에 커밋
        commit_to_lakefs_with_git(
            client=client,
            message=f"Upload model and training data (Accuracy: {metrics['accuracy']:.2f}%)",
            metadata={
                "source": "train.py",
                "uploaded_files": ", ".join(uploaded_files),
                "model_accuracy": f"{metrics['accuracy']:.2f}",
                "model_loss": f"{metrics['loss']:.2f}"
            }
        )
        print("\n=== 변경사항 업로드 완료 ===")
        print("업로드된 파일:")
        for file in uploaded_files:
            print(f"- {file}")
        print(f"\nLakeFS에 모델이 등록되었습니다. 파일명:{MODEL_FILENAME}")
    else:
        print("\n=== 변경사항 없음 ===")
        print("모든 파일이 최신 상태입니다.")
    
    # MLflow 실험 기록
    mlflow_train = MLflowTrain()
    with mlflow.start_run():
        # LakeFS 경로
        lakefs_model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth"
        lakefs_images_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/data/images"
        lakefs_masks_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/data/masks"
        
        # 메트릭과 파라미터 기록
        parameters = {
            "model_path": lakefs_model_path,
            "images_path": lakefs_images_path,
            "masks_path": lakefs_masks_path,
            "epochs": model_train.epochs,
            "learning_rate": model_train.learning_rate
        }
        mlflow_train.log_metrics(metrics)
        mlflow_train.log_params(parameters)
        
        # 모델 및 데이터 경로 출력
        print(f"\n모델 경로: {lakefs_model_path}")
        print(f"이미지 경로: {lakefs_images_path}")
        print(f"마스크 경로: {lakefs_masks_path}")

if __name__ == "__main__":
    main()
