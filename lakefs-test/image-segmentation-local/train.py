from utils.dir_utils import ensure_directories
from train.model_train import ModelTrain
from train.mlflow_train import MLflowTrain
from config import (
    LAKEFS_REPO_NAME, LAKEFS_BRANCH,
    MODEL_PATH, IMAGES_DIR, MASKS_DIR,
    LAKEFS_MODEL_FILE_PATH, LAKEFS_IMAGES_PATH, LAKEFS_MASKS_PATH
)
from utils.lakefs_utils import setup_lakefs_client, upload_to_lakefs, commit_to_lakefs_with_git, get_latest_commit
import mlflow
import os
import git

def get_git_commit_hash():
    """Git 커밋 해시를 가져옵니다."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None

def upload_file_to_lakefs(client, local_path, lakefs_path, uploaded_files):
    """파일을 LakeFS에 업로드하고 변경 여부를 반환합니다."""
    if upload_to_lakefs(client, local_path, lakefs_path):
        display_path = f"{lakefs_path}/*" if os.path.isdir(local_path) else lakefs_path
        uploaded_files.append(display_path)
        return True
    return False

def train_model():
    """모델을 학습하고 메트릭을 반환합니다."""
    model_train = ModelTrain()
    model_train.train()
    return model_train.get_metrics(), model_train

def upload_to_lakefs_storage(client, metrics):
    """모델과 데이터를 LakeFS에 업로드하고 커밋 정보를 반환합니다."""
    uploaded_files = []
    changes_detected = False
    
    upload_targets = [
        (MODEL_PATH, LAKEFS_MODEL_FILE_PATH),
        (IMAGES_DIR, LAKEFS_IMAGES_PATH),
        (MASKS_DIR, LAKEFS_MASKS_PATH)
    ]
    
    for local_path, lakefs_path in upload_targets:
        if upload_file_to_lakefs(client, local_path, lakefs_path, uploaded_files):
            changes_detected = True
    
    commit_info = None
    if changes_detected:
        commit_info = commit_to_lakefs_with_git(
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
    else:
        print("\n=== 변경사항 없음 ===")
        print("모든 파일이 최신 상태입니다.")
        commit_info = get_latest_commit(client)
    
    return commit_info, uploaded_files

def log_to_mlflow(metrics, model_train, commit_info):
    """학습 결과를 MLflow에 기록합니다."""
    mlflow_train = MLflowTrain()
    with mlflow.start_run():
        # LakeFS 경로 설정
        lakefs_model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_MODEL_FILE_PATH}"
        lakefs_images_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_IMAGES_PATH}"
        lakefs_masks_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_MASKS_PATH}"
        
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
        
        # Git과 LakeFS 태그 기록
        tags = {}
        git_commit = get_git_commit_hash()
        if git_commit:
            tags["git_commit_hash"] = git_commit
        
        if commit_info and 'id' in commit_info:
            tags["lakefs_commit_hash"] = commit_info['id']
        
        mlflow_train.log_tags(tags)
        
        # 결과 출력
        print("\n=== MLflow 태그 기록 완료 ===")
        print(f"Git 커밋: {git_commit}")
        print(f"LakeFS 커밋: {commit_info.get('id', 'N/A') if commit_info else 'N/A'}")
        
        print(f"\n모델 경로: {lakefs_model_path}")
        print(f"이미지 경로: {lakefs_images_path}")
        print(f"마스크 경로: {lakefs_masks_path}")

def main():
    """메인 함수"""
    # 필요한 디렉토리 생성
    ensure_directories()
    
    # LakeFS 클라이언트 설정
    client = setup_lakefs_client(create_if_not_exists=True)
    
    # 모델 학습
    metrics, model_train = train_model()
    
    # LakeFS에 업로드
    commit_info, _ = upload_to_lakefs_storage(client, metrics)
    
    # MLflow에 기록
    log_to_mlflow(metrics, model_train, commit_info)

if __name__ == "__main__":
    main()