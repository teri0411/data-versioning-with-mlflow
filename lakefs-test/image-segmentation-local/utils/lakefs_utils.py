import os
import git
import hashlib
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from config import *

def get_git_info():
    """Git 커밋 해시와 브랜치 정보를 가져옵니다."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            'commit_hash': repo.head.commit.hexsha,
            'branch': repo.active_branch.name
        }
    except git.exc.InvalidGitRepositoryError:
        return {
            'commit_hash': None,
            'branch': None
        }

def setup_lakefs_client(create_if_not_exists=False):
    """LakeFS 클라이언트를 설정합니다."""
    configuration = lakefs_client.Configuration()
    configuration.host = LAKEFS_ENDPOINT
    configuration.username = LAKEFS_ACCESS_KEY
    configuration.password = LAKEFS_SECRET_KEY
    client = LakeFSClient(configuration)
    
    if create_if_not_exists:
        try:
            # 저장소가 없으면 생성
            client.repositories_api.create_repository(
                repository_creation=dict(
                    name=LAKEFS_REPO_NAME,
                    storage_namespace=f"s3://{LAKEFS_REPO_NAME}",
                    default_branch=LAKEFS_BRANCH
                )
            )
            print(f"저장소 생성됨: {LAKEFS_REPO_NAME}")
        except Exception as e:
            # 이미 존재하는 경우 무시
            if not ("already exists" in str(e).lower() or "not unique" in str(e).lower()):
                raise e
            print(f"저장소가 이미 존재함: {LAKEFS_REPO_NAME}")
    
    return client

def commit_to_lakefs_with_git(client, message, metadata=None):
    """Git 정보를 포함하여 LakeFS에 커밋합니다."""
    if metadata is None:
        metadata = {}
    # Git 정보 가져오기
    git_info = get_git_info()
    
    # 메타데이터에 Git 정보 추가
    if git_info['commit_hash']:
        metadata['git_commit_hash'] = git_info['commit_hash']
    if git_info['branch']:
        metadata['git_branch'] = git_info['branch']
    
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
        if git_info['commit_hash']:
            print(f"Git 커밋: {git_info['commit_hash']}")
        
        return commit_info
    except Exception as e:
        print(f"LakeFS 커밋 중 오류 발생: {str(e)}")
        return None

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
        stat = client.objects_api.stat_object(
            repository=LAKEFS_REPO_NAME,
            ref=LAKEFS_BRANCH,
            path=lakefs_path
        )
        return stat.checksum
    except lakefs_client.exceptions.NotFoundException:
        return None

def upload_to_lakefs(client, local_path, lakefs_path):
    """파일을 LakeFS에 업로드합니다."""
    try:
        if os.path.isfile(local_path):
            # 단일 파일 업로드
            with open(local_path, 'rb') as f:
                client.objects_api.upload_object(
                    repository=LAKEFS_REPO_NAME,
                    branch=LAKEFS_BRANCH,
                    path=lakefs_path,
                    content=f
                )
            return True
        else:
            print(f"파일을 찾을 수 없음: {local_path}")
            return False
    except Exception as e:
        print(f"업로드 중 오류 발생: {str(e)}")
        return False

def download_from_lakefs(client, lakefs_path, local_path):
    """LakeFS에서 파일을 다운로드합니다."""
    try:
        # 디렉토리 경로가 있으면 생성
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # LakeFS에서 파일 목록 가져오기
        try:
            objects = client.objects_api.list_objects(
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
                response = client.objects_api.get_object(
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
                    response = client.objects_api.get_object(
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
