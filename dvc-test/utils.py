import subprocess
from typing import Dict, Optional

def get_git_commit_info() -> Optional[str]:
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def get_dvc_paths() -> Dict[str, str]:
    """Get DVC tracked file paths"""
    return {
        'data_path': 'data/wine-quality.csv',
        'model_path': 'models/model.pt'
    }
