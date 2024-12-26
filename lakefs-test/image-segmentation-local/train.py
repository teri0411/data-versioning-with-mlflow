from utils.dir_utils import ensure_directories
from train.model_train import ModelTrain
from train.mlflow_train import MLflowTrain
from config import (
    LAKEFS_REPO_NAME, LAKEFS_BRANCH,
    MODEL_PATH, IMAGES_DIR, MASKS_DIR,
    LAKEFS_MODEL_FILE_PATH, LAKEFS_IMAGES_PATH, LAKEFS_MASKS_PATH
)
from utils.lakefs_utils import setup_lakefs_client, upload_to_lakefs, commit_to_lakefs_with_git, get_latest_commit
import mlflow
import os
import git

def get_git_commit_hash():
    """Get Git commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None

def upload_file_to_lakefs(client, local_path, lakefs_path, uploaded_files):
    """Upload file to LakeFS and return if changed."""
    if upload_to_lakefs(client, local_path, lakefs_path):
        display_path = f"{lakefs_path}/*" if os.path.isdir(local_path) else lakefs_path
        uploaded_files.append(display_path)
        return True
    return False

def train_model():
    """Train model and return metrics."""
    model_train = ModelTrain()
    model_train.train()
    return model_train.get_metrics(), model_train

def upload_to_lakefs_storage(client, metrics):
    """Upload model and data to LakeFS and return commit info."""
    uploaded_files = []
    changes_detected = False
    
    upload_targets = [
        (MODEL_PATH, LAKEFS_MODEL_FILE_PATH),
        (IMAGES_DIR, LAKEFS_IMAGES_PATH),
        (MASKS_DIR, LAKEFS_MASKS_PATH)
    ]
    
    for local_path, lakefs_path in upload_targets:
        if upload_file_to_lakefs(client, local_path, lakefs_path, uploaded_files):
            changes_detected = True
    
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
        print("\n=== Upload Changes Complete ===")
        print("Uploaded files:")
        for file in uploaded_files:
            print(f"- {file}")
    else:
        print("\n=== No Changes ===")
        print("All files are up to date.")
        commit_info = get_latest_commit(client)
    
    return commit_info, uploaded_files

def log_to_mlflow(metrics, model_train, commit_info):
    """Record training results in MLflow."""
    mlflow_train = MLflowTrain()
    with mlflow.start_run():
        # Set LakeFS paths
        lakefs_model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_MODEL_FILE_PATH}"
        lakefs_images_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_IMAGES_PATH}"
        lakefs_masks_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{LAKEFS_MASKS_PATH}"
        
        # Record metrics and parameters
        parameters = {
            "model_path": lakefs_model_path,
            "images_path": lakefs_images_path,
            "masks_path": lakefs_masks_path,
            "epochs": model_train.epochs,
            "learning_rate": model_train.learning_rate
        }
        mlflow_train.log_metrics(metrics)
        mlflow_train.log_params(parameters)
        
        # Record Git and LakeFS tags
        tags = {}
        git_commit = get_git_commit_hash()
        if git_commit:
            tags["git_commit_hash"] = git_commit
        
        if commit_info and 'id' in commit_info:
            tags["lakefs_commit_hash"] = commit_info['id']
        
        mlflow_train.log_tags(tags)
        
        # Output results
        print("\n=== MLflow Tags Recorded ===")
        print(f"Git Commit: {git_commit}")
        print(f"LakeFS Commit: {commit_info.get('id', 'N/A') if commit_info else 'N/A'}")
        
        print(f"\nModel Path: {lakefs_model_path}")
        print(f"Images Path: {lakefs_images_path}")
        print(f"Masks Path: {lakefs_masks_path}")

def main():
    """Main function"""
    # Create necessary directories
    ensure_directories()
    
    # Set up LakeFS client
    client = setup_lakefs_client(create_if_not_exists=True)
    
    # Train model
    metrics, model_train = train_model()
    
    # Upload to LakeFS
    commit_info, _ = upload_to_lakefs_storage(client, metrics)
    
    # Record in MLflow
    log_to_mlflow(metrics, model_train, commit_info)

if __name__ == "__main__":
    main()