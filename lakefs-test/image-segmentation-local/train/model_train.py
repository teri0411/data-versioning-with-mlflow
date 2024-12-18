from .base_train import BaseTrain
from .mlflow_train import MLflowTrain
from .lakefs_train import LakeFSTrain
from config import EPOCHS

class ModelTrain:
    """전체 학습 과정을 관리하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.base_train = BaseTrain()
        self.mlflow_train = MLflowTrain()
        self.lakefs_train = LakeFSTrain()
    
    def train(self):
        """학습을 수행합니다."""
        # MLflow 실험 시작
        with self.mlflow_train.start_run():
            # 학습 파라미터 기록
            self.mlflow_train.log_params()
            
            # 모델 학습
            for epoch in range(EPOCHS):
                loss = self.base_train.train_epoch(epoch)
                self.mlflow_train.log_metrics({"loss": loss})
            
            # 모델 저장
            self.lakefs_train.save_model(self.base_train.model)
            
            # 데이터 업로드
            self.lakefs_train.upload_data()
