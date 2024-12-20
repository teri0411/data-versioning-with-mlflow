import os
import hashlib
import subprocess
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from config import *

def get_git_hash():
    """현재 Git 커밋 해시를 가져옵니다."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def get_git_branch():
    """현재 Git 브랜치 이름을 가져옵니다."""
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                              capture_output=True,
                              text=True,
                              check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def commit_to_lakefs_with_git(client, message, metadata=None):
    """Git 정보를 포함하여 LakeFS에 커밋합니다."""
    if metadata is None:
        metadata = {}
    # Git 정보 가져오기
    git_hash = get_git_hash()
    git_branch = get_git_branch()
    
    # 메타데이터에 Git 정보 추가
    if git_hash:
        metadata['git_commit_hash'] = git_hash
    if git_branch:
        metadata['git_branch'] = git_branch
    
    # LakeFS 커밋 생성
    commit_creation = models.CommitCreation(
        message=message,
        metadata=metadata
    )
    
    try:
        response = client.commits_api.commit(
            repository=LAKEFS_REPO_NAME,
            branch=LAKEFS_BRANCH,
            commit_creation=commit_creation
        )
        
        # response를 딕셔너리로 변환
        commit_info = {
            'id': response.id,
            'message': response.message,
            'metadata': response.metadata
        }
        
        print("\n=== LakeFS 커밋 완료 ===")
        print(f"커밋 ID: {commit_info['id']}")
        if git_hash:
            print(f"Git 커밋: {git_hash}")
        
        return commit_info
    except Exception as e:
        print(f"LakeFS 커밋 중 오류 발생: {str(e)}")
        return None

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

def calculate_file_hash(file_path):
    """파일의 MD5 해시를 계산합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_object_checksum(client, lakefs_path):
    """LakeFS에서 파일의 체크섬을 가져옵니다."""
    try:
        stat = client.objects.stat_object(
            repository=LAKEFS_REPO_NAME,
            ref=LAKEFS_BRANCH,
            path=lakefs_path
        )
        return stat.checksum
    except lakefs_client.exceptions.NotFoundException:
        return None

def upload_to_lakefs(client, local_path, lakefs_path):
    """파일이나 디렉토리를 LakeFS에 업로드합니다.
    LakeFS의 체크섬 기준으로 변경사항을 감지합니다.
    """
    try:
        # 디렉토리인 경우
        if os.path.isdir(local_path):
            any_uploaded = False
            for filename in os.listdir(local_path):
                local_file = os.path.join(local_path, filename)
                lakefs_file = os.path.join(lakefs_path, filename).replace('\\', '/')
                if upload_to_lakefs(client, local_file, lakefs_file):
                    any_uploaded = True
            return any_uploaded
        
        # 파일인 경우
        else:
            if not os.path.exists(local_path):
                print(f"Error: File not found: {local_path}")
                return False
            
            try:
                # 현재 파일의 체크섬 계산
                current_hash = calculate_file_hash(local_path)
                
                # LakeFS의 파일 체크섬 가져오기
                lakefs_checksum = get_object_checksum(client, lakefs_path)
                
                # 체크섬 비교
                if lakefs_checksum and lakefs_checksum == current_hash:
                    print(f"파일 변경 없음, 스킵: {lakefs_path}")
                    return False
                
                # 파일이 변경되었거나 새로운 파일인 경우 업로드
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
                
    except Exception as e:
        print(f"Error: {str(e)}")
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

def get_latest_commit(client):
    """가장 최근의 LakeFS 커밋 정보를 가져옵니다."""
    try:
        # 브랜치의 최신 커밋 가져오기
        branch_info = client.branches_api.get_branch(
            repository=LAKEFS_REPO_NAME,
            branch=LAKEFS_BRANCH
        )
        return {
            'id': branch_info.commit_id,
            'metadata': {}  # metadata가 없으면 빈 딕셔너리 반환
        }
    except Exception as e:
        print(f"LakeFS 커밋 정보 조회 중 오류 발생: {str(e)}")
        return None
