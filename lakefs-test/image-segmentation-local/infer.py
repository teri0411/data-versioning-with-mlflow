import os
import mlflow
import torch
from PIL import Image
import torchvision.transforms as transforms

from config import *
from utils import setup_lakefs_client, download_from_lakefs
from model import create_model

class Inferencer:
    """모델 추론을 관리하는 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        ])
        self.lakefs_client = setup_lakefs_client()
        
        # MLflow 설정
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def load_run(self, run_id=None):
        """MLflow에서 실험 정보를 가져옵니다."""
        print("MLflow에서 실험 정보 가져오기...")
        
        if run_id:
            run = mlflow.get_run(run_id)
            print(f"- Run ID: {run_id}")
        else:
            print("- Run ID가 지정되지 않아 최신 실험을 가져옵니다")
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if len(runs) == 0:
                raise Exception("MLflow에서 실험을 찾을 수 없습니다.")
            run = mlflow.get_run(runs.iloc[0].run_id)
        
        return run
    
    def load_model(self, model_path):
        """LakeFS에서 모델을 다운로드하고 로드합니다."""
        if model_path.startswith(f"lakefs://{LAKEFS_REPO_NAME}/"):
            # LakeFS 경로에서 실제 경로 추출
            lakefs_path = model_path.replace(f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/", "")
            if not download_from_lakefs(self.lakefs_client, lakefs_path, MODEL_PATH):
                raise Exception("Failed to download model from LakeFS")
        
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
    
    def infer_image(self, image_path):
        """단일 이미지에 대한 추론을 수행합니다."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        return output.squeeze().cpu()
    
    def run_inference(self, run_id=None):
        """전체 추론 과정을 실행합니다."""
        # MLflow에서 실험 정보 가져오기
        run = self.load_run(run_id)
        
        # 메트릭과 태그 가져오기
        metrics = run.data.metrics
        tags = run.data.tags
        
        # Git 커밋 해시 가져오기
        git_commit = tags.get("git_commit_hash", "N/A")
        
        # 메트릭 가져오기
        loss = metrics.get("Loss", "N/A")
        best_loss = metrics.get("Best Loss", "N/A")
        final_loss = metrics.get("Final Loss", "N/A")
        
        # 숫자 형식화
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        if isinstance(best_loss, float):
            best_loss = f"{best_loss:.4f}"
        if isinstance(final_loss, float):
            final_loss = f"{final_loss:.4f}"
        
        print(f"\nRun ID: {run.info.run_id}")
        print(f"- Git Commit: {git_commit}")
        print(f"- Loss: {loss}")
        print(f"- Best Loss: {best_loss}")
        print(f"- Final Loss: {final_loss}")
        
        # 모델 로드
        model_path = run.data.params.get("model_path")
        if not model_path:
            raise Exception("모델 경로를 찾을 수 없습니다.")
        
        print("\n모델 로드 중...")
        self.load_model(model_path)
        print("모델 로드 완료!")
        
        return run

def main():
    """메인 함수"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="MLflow run ID")
    args = parser.parse_args()
    
    inferencer = Inferencer()
    inferencer.run_inference(args.run_id)

if __name__ == "__main__":
    main()
