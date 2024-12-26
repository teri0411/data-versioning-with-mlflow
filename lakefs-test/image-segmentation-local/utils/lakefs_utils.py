import os
import git
import hashlib
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from config import *

def get_git_info():
    """Get Git commit hash and branch information."""
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
    """Set up LakeFS client."""
    configuration = lakefs_client.Configuration()
    configuration.host = LAKEFS_ENDPOINT
    configuration.username = LAKEFS_ACCESS_KEY
    configuration.password = LAKEFS_SECRET_KEY
    client = LakeFSClient(configuration)
    
    if create_if_not_exists:
        try:
            # Create repository if it doesn't exist
            client.repositories_api.create_repository(
                repository_creation=dict(
                    name=LAKEFS_REPO_NAME,
                    storage_namespace=f"s3://{LAKEFS_REPO_NAME}",
                    default_branch=LAKEFS_BRANCH
                )
            )
            print(f"Repository created: {LAKEFS_REPO_NAME}")
        except Exception as e:
            # Ignore if already exists
            if not ("already exists" in str(e).lower() or "not unique" in str(e).lower()):
                raise e
            print(f"Repository already exists: {LAKEFS_REPO_NAME}")
    
    return client

def commit_to_lakefs_with_git(client, message, metadata=None):
    """Commit to LakeFS with Git information."""
    if metadata is None:
        metadata = {}
    # Get Git information
    git_info = get_git_info()
    
    # Add Git information to metadata
    if git_info['commit_hash']:
        metadata['git_commit_hash'] = git_info['commit_hash']
    if git_info['branch']:
        metadata['git_branch'] = git_info['branch']
    
    # Create LakeFS commit
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
        
        # Convert response to dictionary
        commit_info = {
            'id': response.id,
            'message': response.message,
            'metadata': response.metadata
        }
        
        print("\n=== LakeFS Commit Complete ===")
        print(f"Commit ID: {commit_info['id']}")
        if git_info['commit_hash']:
            print(f"Git Commit: {git_info['commit_hash']}")
        
        return commit_info
    except Exception as e:
        print(f"Error during LakeFS commit: {str(e)}")
        return None

def calculate_file_hash(file_path):
    """Calculate MD5 hash of the file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_object_checksum(client, lakefs_path):
    """Get file checksum from LakeFS."""
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
    """Upload file to LakeFS."""
    try:
        if os.path.isfile(local_path):
            # Upload single file
            with open(local_path, 'rb') as f:
                client.objects_api.upload_object(
                    repository=LAKEFS_REPO_NAME,
                    branch=LAKEFS_BRANCH,
                    path=lakefs_path,
                    content=f
                )
            return True
        else:
            print(f"File not found: {local_path}")
            return False
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return False

def download_from_lakefs(client, lakefs_path, local_path):
    """Download file from LakeFS."""
    try:
        # Create directory path if it exists
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Get file list from LakeFS
        try:
            objects = client.objects_api.list_objects(
                repository=LAKEFS_REPO_NAME,
                ref=LAKEFS_BRANCH,
                prefix=lakefs_path
            )
            
            # Fail if file doesn't exist
            if not objects.results:
                print(f"File not found in LakeFS: {lakefs_path}")
                return False
                
            # For single file
            if not os.path.isdir(local_path):
                response = client.objects_api.get_object(
                    repository=LAKEFS_REPO_NAME,
                    ref=LAKEFS_BRANCH,
                    path=lakefs_path
                )
                
                # First save to temporary file
                temp_path = local_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    f.write(response.read())
                    
                # Delete existing file if it exists
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except OSError:
                        os.chmod(local_path, 0o777)
                        os.remove(local_path)
                        
                # Move temporary file to actual path
                os.replace(temp_path, local_path)
                return True
                
            # For directory
            else:
                success = True
                for obj in objects.results:
                    # Create file path
                    rel_path = obj.path.replace(lakefs_path, '').lstrip('/')
                    dest_path = os.path.join(local_path, rel_path)
                    
                    # Create directory
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Download file
                    response = client.objects_api.get_object(
                        repository=LAKEFS_REPO_NAME,
                        ref=LAKEFS_BRANCH,
                        path=obj.path
                    )
                    
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                        
                return success
                
        except Exception as e:
            print(f"LakeFS API call failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Failed to download from LakeFS ({lakefs_path}): {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def get_latest_commit(client):
    """Get the most recent LakeFS commit information."""
    try:
        # Get latest commit from branch
        branch_info = client.branches_api.get_branch(
            repository=LAKEFS_REPO_NAME,
            branch=LAKEFS_BRANCH
        )
        return {
            'id': branch_info.commit_id,
            'metadata': {}  # Return empty dictionary if no metadata
        }
    except Exception as e:
        print(f"Error while retrieving LakeFS commit information: {str(e)}")
        return None
