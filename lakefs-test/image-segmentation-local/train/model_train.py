import os
import torch
from .base_train import BaseTrain
from config import EPOCHS, MODEL_DIR, MODEL_PATH

class ModelTrain:
    """모델 학습을 관리하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.base_train = BaseTrain()
        self.loss = 0.0
        self.accuracy = 0.0
        self.epochs = EPOCHS
        self.learning_rate = 0.001  # BaseTrain의 learning rate와 동일하게 설정
    
    def save_model(self, model):
        """모델을 파일로 저장합니다."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    def train(self):
        """
        학습을 수행하고 학습된 모델을 반환합니다.
        
        Returns:
            학습된 모델
        """
        # 모델 학습
        for epoch in range(self.epochs):
            loss = self.base_train.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")
            self.loss = loss  # 마지막 epoch의 loss 저장
        
        # 정확도 계산 (예시)
        self.accuracy = 0.85  # 실제로는 검증 데이터로 계산해야 함
        
        # 모델 저장
        self.save_model(self.base_train.model)
        
        return self.base_train.model
