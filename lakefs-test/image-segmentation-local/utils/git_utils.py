import git

def get_git_commit_hash():
    """Return the current Git commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None
