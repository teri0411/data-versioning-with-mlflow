from utils.dir_utils import ensure_directories
from train.model_train import ModelTrain
from train.mlflow_train import MLflowTrain
from config import LAKEFS_REPO_NAME, LAKEFS_BRANCH, MODEL_FILENAME
from utils.lakefs_utils import setup_lakefs_client, upload_to_lakefs, commit_to_lakefs_with_git, get_latest_commit
import mlflow
import os
import git

def get_git_commit_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None

def upload_file_to_lakefs(client, local_path, lakefs_path, uploaded_files):
    """파일을 LakeFS에 업로드하고 변경 여부를 반환합니다."""
    if upload_to_lakefs(client, local_path, lakefs_path):
        # 디렉토리인 경우 와일드카드(*) 추가
        display_path = f"{lakefs_path}/*" if os.path.isdir(local_path) else lakefs_path
        uploaded_files.append(display_path)
        return True
    return False

def main():
    # 필요한 디렉토리 생성
    ensure_directories()
    
    # LakeFS 클라이언트 설정
    client = setup_lakefs_client(create_if_not_exists=True)
    
    # 모델 학습
    model_train = ModelTrain()
    model_train.train()  # 모델 학습 실행
    metrics = model_train.get_metrics()  # 메트릭 가져오기
    
    # 절대 경로 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/model.pth")
    images_path = os.path.join(current_dir, "data/images")
    masks_path = os.path.join(current_dir, "data/masks")

    # LakeFS에 파일 업로드
    uploaded_files = []
    changes_detected = False
    
    # 파일 업로드 실행
    upload_targets = [
        (model_path, "models/model.pth"),
        (images_path, "data/images"),
        (masks_path, "data/masks")
    ]
    
    for local_path, lakefs_path in upload_targets:
        if upload_file_to_lakefs(client, local_path, lakefs_path, uploaded_files):
            changes_detected = True
    
    # 변경사항이 있는 경우 LakeFS에 커밋
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
        # 변경사항이 없을 경우 최신 커밋 정보 가져오기
        commit_info = get_latest_commit(client)

    # MLflow 실험 기록 (변경사항 여부와 관계없이 항상 실행)
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
        
        # Git과 LakeFS 태그 기록
        tags = {}
        git_commit = get_git_commit_hash()
        if git_commit:
            tags["git_commit_hash"] = git_commit
        
        if commit_info and 'id' in commit_info:
            tags["lakefs_commit_hash"] = commit_info['id']
        
        mlflow_train.log_tags(tags)
        
        print("\n=== MLflow 태그 기록 완료 ===")
        print(f"Git 커밋: {git_commit}")
        print(f"LakeFS 커밋: {commit_info.get('id', 'N/A') if commit_info else 'N/A'}")
        
        # 모델 및 데이터 경로 출력
        print(f"\n모델 경로: {lakefs_model_path}")
        print(f"이미지 경로: {lakefs_images_path}")
        print(f"마스크 경로: {lakefs_masks_path}")

if __name__ == "__main__":
    main()