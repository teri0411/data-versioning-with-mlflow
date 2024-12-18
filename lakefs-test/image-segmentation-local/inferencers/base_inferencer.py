import torch
import torchvision.transforms as transforms
from PIL import Image
from config import *
from model import create_model

class BaseInferencer:
    """기본 모델 추론 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        ])
    
    def load_model_weights(self, model_path):
        """모델 가중치를 로드합니다."""
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def infer_image(self, image_path):
        """단일 이미지에 대한 추론을 수행합니다."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        return output.squeeze().cpu()
