import os
import torch
import numpy as np
from PIL import Image
from config import *
from model import create_model

class BaseInference:
    """기본 모델 추론 클래스"""
    
    def __init__(self):
        """초기화"""
        self.model = create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def load_model(self, model_path):
        """모델을 로드합니다."""
        print(f"\n모델 로드 중... ({model_path})")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def infer_images(self):
        """모든 이미지에 대해 추론을 수행합니다."""
        results = []
        images_dir = os.path.join(DATA_PATH, "images")
        
        print("\n추론 시작...")
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(".png"):
                continue
                
            image_path = os.path.join(images_dir, image_file)
            output = self.infer_image(image_path)
            results.append((image_file, output))
            print(f"이미지 처리 완료: {image_file}")
            
        return results
    
    def infer_image(self, image_path):
        """단일 이미지에 대해 추론을 수행합니다."""
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image) / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        # 추론 수행
        with torch.no_grad():
            output = self.model(image)
            
        # 후처리
        output = torch.sigmoid(output).cpu().numpy()
        output = (output > 0.5).astype(np.uint8) * 255
        return output
    
    def save_results(self, results):
        """추론 결과를 저장합니다."""
        print("\n결과 저장 중...")
        output_dir = os.path.join(DATA_PATH, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        for image_file, output in results:
            output_image = Image.fromarray(output.squeeze())
            output_path = os.path.join(output_dir, image_file.replace("image", "pred"))
            output_image.save(output_path)
            
        print(f"결과가 {output_dir}에 저장되었습니다.")
