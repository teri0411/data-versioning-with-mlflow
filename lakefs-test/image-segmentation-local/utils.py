import os
import subprocess
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from config import *

def get_git_commit_hash():
    """현재 Git commit hash를 반환합니다."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return None

def setup_lakefs_client():
    """LakeFS 클라이언트를 설정하고 반환합니다."""
    configuration = lakefs_client.Configuration()
    configuration.host = LAKEFS_ENDPOINT
    configuration.username = LAKEFS_ACCESS_KEY
    configuration.password = LAKEFS_SECRET_KEY
    client = LakeFSClient(configuration)
    
    # 저장소가 없으면 생성
    try:
        client.repositories.get_repository(LAKEFS_REPO_NAME)
    except lakefs_client.exceptions.NotFoundException:
        client.repositories.create_repository(
            models.RepositoryCreation(
                name=LAKEFS_REPO_NAME,
                storage_namespace=f"s3://{LAKEFS_REPO_NAME}",
                default_branch=LAKEFS_BRANCH,
            )
        )
        print(f"Repository '{LAKEFS_REPO_NAME}' created successfully!")
    else:
        print(f"Repository '{LAKEFS_REPO_NAME}' already exists.")
    
    return client

def upload_to_lakefs(client, local_path, lakefs_path):
    """파일을 LakeFS에 업로드합니다."""
    try:
        with open(local_path, 'rb') as f:
            client.objects.upload_object(
                repository=LAKEFS_REPO_NAME,
                branch=LAKEFS_BRANCH,
                path=lakefs_path,
                content=f
            )
        return True
    except Exception as e:
        print(f"Error uploading {local_path} to LakeFS: {str(e)}")
        return False

def download_from_lakefs(client, lakefs_path, local_path):
    """LakeFS에서 파일을 다운로드합니다."""
    try:
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        response = client.objects.get_object(
            repository=LAKEFS_REPO_NAME,
            ref=LAKEFS_BRANCH,
            path=lakefs_path
        )
        with open(local_path, 'wb') as f:
            f.write(response.read())
        return True
    except Exception as e:
        print(f"Error downloading {lakefs_path}: {str(e)}")
        return False

def ensure_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 생성합니다."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
