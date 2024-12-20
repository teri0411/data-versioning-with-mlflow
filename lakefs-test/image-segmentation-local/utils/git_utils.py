import git

def get_git_commit_hash():
    """현재 Git 커밋 해시를 반환합니다."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None
