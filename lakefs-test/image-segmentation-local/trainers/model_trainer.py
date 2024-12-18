from .base_trainer import BaseTrainer
from .mlflow_trainer import MLflowTrainer
from .lakefs_trainer import LakeFSTrainer
from config import EPOCHS

class ModelTrainer:
    """전체 학습 과정을 관리하는 클래스"""
    
    def __init__(self):
        self.base_trainer = BaseTrainer()
        self.mlflow_trainer = MLflowTrainer()
        self.lakefs_trainer = LakeFSTrainer()
    
    def train(self):
        """전체 학습 과정을 실행합니다."""
        with self.mlflow_trainer.start_run():
            # 학습 파라미터 기록
            self.mlflow_trainer.log_params()
            
            # 학습 실행
            print("\n학습 시작...")
            best_loss = float('inf')
            final_loss = None
            
            for epoch in range(EPOCHS):
                avg_loss = self.base_trainer.train_epoch(epoch)
                self.mlflow_trainer.log_metrics({"Loss": avg_loss})
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                final_loss = avg_loss
            
            # 최종 메트릭 기록
            self.mlflow_trainer.log_metrics({
                "Best Loss": best_loss,
                "Final Loss": final_loss
            })
            
            # 모델 저장 및 업로드
            model_path = self.lakefs_trainer.save_model(self.base_trainer.model)
            self.mlflow_trainer.log_model_path(model_path)
            
            # 데이터 업로드
            self.lakefs_trainer.upload_data()
            self.mlflow_trainer.log_data_path()
