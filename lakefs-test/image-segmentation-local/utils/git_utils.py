import subprocess

def get_git_commit_hash():
    """현재 Git commit hash를 반환합니다."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return None
