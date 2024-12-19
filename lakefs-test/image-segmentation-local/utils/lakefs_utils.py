import os
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from config import *

def setup_lakefs_client(create_if_not_exists=False):
    """LakeFS 클라이언트를 설정하고 반환합니다."""
    configuration = lakefs_client.Configuration()
    configuration.host = LAKEFS_ENDPOINT
    configuration.username = LAKEFS_ACCESS_KEY
    configuration.password = LAKEFS_SECRET_KEY
    client = LakeFSClient(configuration)
    
    # 저장소가 없으면 생성 (train.py에서만 사용)
    try:
        client.repositories.get_repository(LAKEFS_REPO_NAME)
    except lakefs_client.exceptions.NotFoundException:
        if create_if_not_exists:
            client.repositories.create_repository(
                models.RepositoryCreation(
                    name=LAKEFS_REPO_NAME,
                    storage_namespace=f"s3://{LAKEFS_REPO_NAME}",
                    default_branch=LAKEFS_BRANCH,
                )
            )
            print(f"Repository '{LAKEFS_REPO_NAME}' created successfully!")
        else:
            raise
    
    return client

def upload_to_lakefs(client, local_path, lakefs_path):
    """파일이나 디렉토리를 LakeFS에 업로드합니다."""
    try:
        # 디렉토리인 경우
        if os.path.isdir(local_path):
            success = True
            for filename in os.listdir(local_path):
                local_file = os.path.join(local_path, filename)
                lakefs_file = os.path.join(lakefs_path, filename).replace('\\', '/')
                if not upload_to_lakefs(client, local_file, lakefs_file):
                    success = False
            return success
        
        # 파일인 경우
        else:
            with open(local_path, 'rb') as f:
                client.objects.upload_object(
                    repository=LAKEFS_REPO_NAME,
                    branch=LAKEFS_BRANCH,
                    path=lakefs_path,
                    content=f
                )
            print(f"파일 업로드 완료: {lakefs_path}")
            return True
    except Exception as e:
        print(f"Error uploading {local_path} to LakeFS: {str(e)}")
        return False

def download_from_lakefs(client, lakefs_path, local_path):
    """LakeFS에서 파일을 다운로드합니다."""
    try:
        # 디렉토리 경로가 있으면 생성
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # LakeFS에서 파일 목록 가져오기
        try:
            objects = client.objects.list_objects(
                repository=LAKEFS_REPO_NAME,
                ref=LAKEFS_BRANCH,
                prefix=lakefs_path
            )
            
            # 파일이 없으면 실패
            if not objects.results:
                print(f"LakeFS에서 파일을 찾을 수 없음: {lakefs_path}")
                return False
                
            # 단일 파일인 경우
            if not os.path.isdir(local_path):
                response = client.objects.get_object(
                    repository=LAKEFS_REPO_NAME,
                    ref=LAKEFS_BRANCH,
                    path=lakefs_path
                )
                
                # 임시 파일로 먼저 저장
                temp_path = local_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    f.write(response.read())
                    
                # 기존 파일이 있으면 삭제
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except OSError:
                        os.chmod(local_path, 0o777)
                        os.remove(local_path)
                        
                # 임시 파일을 실제 경로로 이동
                os.replace(temp_path, local_path)
                return True
                
            # 디렉토리인 경우
            else:
                success = True
                for obj in objects.results:
                    # 파일 경로 생성
                    rel_path = obj.path.replace(lakefs_path, '').lstrip('/')
                    dest_path = os.path.join(local_path, rel_path)
                    
                    # 디렉토리 생성
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # 파일 다운로드
                    response = client.objects.get_object(
                        repository=LAKEFS_REPO_NAME,
                        ref=LAKEFS_BRANCH,
                        path=obj.path
                    )
                    
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                        
                return success
                
        except Exception as e:
            print(f"LakeFS API 호출 실패: {str(e)}")
            return False
            
    except Exception as e:
        print(f"LakeFS에서 다운로드 실패 ({lakefs_path}): {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return False
