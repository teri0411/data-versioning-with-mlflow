import os
import mlflow
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from config import *
from utils import setup_lakefs_client, download_from_lakefs
from model import create_model
from dataset import SegmentationDataset
from torch.utils.data import DataLoader

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
    
    def load_run(self):
        """MLflow에서 실험 정보를 가져옵니다."""
        print("MLflow에서 실험 정보 가져오기...\n")
        
        # 실험 목록 가져오기
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if len(runs) == 0:
            print("저장된 실험이 없습니다.")
            return None
        
        print("사용 가능한 실험 목록:\n")
        for idx, run_info in runs.iterrows():
            run = mlflow.get_run(run_info.run_id)
            
            # Git commit hash 가져오기
            git_commit = run.data.tags.get("mlflow.source.git.commit", "N/A")
            if git_commit != "N/A":
                git_commit = git_commit[:8]  # 앞 8자리만 표시
            
            # 파라미터와 메트릭 가져오기
            params = {k: v for k, v in run.data.params.items()}
            metrics = {k: round(float(v), 4) for k, v in run.data.metrics.items()}
            
            print(f"{idx + 1}. Run ID: {run_info.run_id}")
            print(f"   Git Commit: {git_commit}")
            print("   Parameters:")
            for k, v in params.items():
                print(f"   - {k}: {v}")
            print("   Metrics:")
            for k, v in metrics.items():
                print(f"   - {k}: {v}")
            print()
        
        # 실험 선택
        while True:
            try:
                choice = input("사용할 실험 번호를 선택하세요: ")
                if not choice.strip():  # 빈 입력 처리
                    print("실험 선택이 취소되었습니다.")
                    return None
                
                choice = int(choice)
                if 1 <= choice <= len(runs):
                    run_id = runs.iloc[choice - 1].run_id
                    return mlflow.get_run(run_id)
                else:
                    print(f"1부터 {len(runs)}까지의 숫자를 입력해주세요.")
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
            except EOFError:
                print("실험 선택이 취소되었습니다.")
                return None
    
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
    
    def run_inference(self):
        """선택한 실험의 모델로 추론을 실행합니다."""
        print("추론 실행 중...")
        
        # MLflow에서 실험 정보 가져오기
        run = self.load_run()
        if run is None:
            print("실험 선택이 취소되었습니다.")
            return
        
        # Git commit hash 가져오기
        git_commit = run.data.tags.get("mlflow.source.git.commit", "N/A")
        if git_commit != "N/A":
            git_commit = git_commit[:8]  # 앞 8자리만 표시
        
        # 파라미터와 메트릭 가져오기
        params = {k: v for k, v in run.data.params.items()}
        metrics = {k: round(float(v), 4) for k, v in run.data.metrics.items()}
        
        print("\n선택한 실험 정보:")
        print(f"- Run ID: {run.info.run_id}")
        print(f"- Git Commit: {git_commit}")
        print("  Parameters:")
        for k, v in params.items():
            print(f"  - {k}: {v}")
        print("  Metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v}")
        
        # 모델 로드
        model_path = params.get("model_path", params.get("model_lakefs_path", params.get("lakefs_model_path")))
        if not model_path:
            print("모델 경로를 찾을 수 없습니다.")
            return
        
        print(f"\n모델 로드 중... ({model_path})")
        self.load_model(model_path)
        
        # 데이터 로드
        data_path = params.get("lakefs_data_path", params.get("data_path"))
        if not data_path:
            print("데이터 경로를 찾을 수 없습니다.")
            return
        
        print(f"데이터 로드 중... ({data_path})")
        
        # LakeFS에서 데이터 다운로드
        local_data_dir = os.path.join("data", run.info.run_id)
        os.makedirs(local_data_dir, exist_ok=True)
        
        # 이미지와 마스크 디렉토리 생성
        local_images_dir = os.path.join(local_data_dir, "images")
        local_masks_dir = os.path.join(local_data_dir, "masks")
        os.makedirs(local_images_dir, exist_ok=True)
        os.makedirs(local_masks_dir, exist_ok=True)
        
        # LakeFS에서 데이터 다운로드
        lakefs_path = data_path.replace(f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/", "")
        lakefs_images_dir = os.path.join(lakefs_path, "images")
        lakefs_masks_dir = os.path.join(lakefs_path, "masks")
        
        print(f"LakeFS에서 이미지 다운로드 중... ({lakefs_images_dir})")
        for image_file in ["image_0.png", "image_1.png", "image_2.png", "image_3.png", "image_4.png"]:
            lakefs_path = os.path.join(lakefs_images_dir, image_file)
            local_path = os.path.join(local_images_dir, image_file)
            download_from_lakefs(self.lakefs_client, lakefs_path, local_path)
        
        print(f"LakeFS에서 마스크 다운로드 중... ({lakefs_masks_dir})")
        for mask_file in ["mask_0.png", "mask_1.png", "mask_2.png", "mask_3.png", "mask_4.png"]:
            lakefs_path = os.path.join(lakefs_masks_dir, mask_file)
            local_path = os.path.join(local_masks_dir, mask_file)
            download_from_lakefs(self.lakefs_client, lakefs_path, local_path)
        
        dataset = SegmentationDataset(local_images_dir, local_masks_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 결과 저장 디렉토리 생성
        output_dir = os.path.join("outputs", run.info.run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 추론 실행
        print("\n추론 실행 중...")
        with torch.no_grad():
            for i, (images, masks) in enumerate(dataloader):
                outputs = self.model(images.to(self.device))
                predicted = (outputs > 0.5).float()
                
                # 결과 시각화
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 원본 이미지 (정규화 해제)
                img = images[0].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                
                # 실제 마스크
                axes[1].imshow(masks[0].squeeze().cpu().numpy(), cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                
                # 예측 마스크
                axes[2].imshow(predicted[0].squeeze().cpu().numpy(), cmap="gray")
                axes[2].set_title("Prediction")
                axes[2].axis("off")
                
                # 결과 저장
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"result_{i+1}.png"))
                plt.close()
                
                print(f"이미지 {i + 1}: 예측 완료 (outputs/{run.info.run_id}/result_{i+1}.png에 저장됨)")
        
        print(f"\n추론이 완료되었습니다. 결과는 {output_dir} 디렉토리에 저장되었습니다.")

def main():
    """메인 함수"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    inferencer = Inferencer()
    inferencer.run_inference()

if __name__ == "__main__":
    main()
