import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from utils import get_git_commit_hash, setup_lakefs_client, upload_to_lakefs, ensure_directories
from dataset import get_data_loader
from model import create_model

class Trainer:
    """모델 학습을 관리하는 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.dataloader = get_data_loader()
        self.lakefs_client = setup_lakefs_client()
    
    def train_epoch(self, epoch):
        """한 에폭 동안의 학습을 수행합니다."""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for images, masks in self.dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_model(self):
        """모델을 저장하고 LakeFS에 업로드합니다."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        print(f"\n모델 저장 완료: {MODEL_PATH}")
        
        # LakeFS에 모델 업로드
        model_lakefs_path = f"models/{MODEL_FILENAME}"
        upload_to_lakefs(self.lakefs_client, MODEL_PATH, model_lakefs_path)
        return f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/{model_lakefs_path}"
    
    def upload_data(self):
        """학습 데이터를 LakeFS에 업로드합니다."""
        print("- 데이터 파일 업로드:")
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                lakefs_path = os.path.relpath(local_path)
                print(f"  - {lakefs_path}")
                upload_to_lakefs(self.lakefs_client, local_path, lakefs_path)
    
    def train(self):
        """전체 학습 과정을 실행합니다."""
        # MLflow 설정
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run() as run:
            # Git commit hash 기록
            git_commit_hash = get_git_commit_hash()
            if git_commit_hash:
                mlflow.set_tag("git_commit_hash", git_commit_hash)
            
            # 학습 파라미터 기록
            mlflow.log_params({
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS
            })
            
            # 학습 실행
            print("\n학습 시작...")
            best_loss = float('inf')
            final_loss = None
            
            for epoch in range(EPOCHS):
                avg_loss = self.train_epoch(epoch)
                mlflow.log_metric("Loss", avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                final_loss = avg_loss
            
            # 최종 메트릭 기록
            mlflow.log_metric("Best Loss", best_loss)
            mlflow.log_metric("Final Loss", final_loss)
            
            # 모델 저장 및 업로드
            model_path = self.save_model()
            mlflow.log_param("model_path", model_path)
            
            # 데이터 업로드
            self.upload_data()
            mlflow.log_param("data_path", f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/data")

def main():
    """메인 함수"""
    ensure_directories()
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()
