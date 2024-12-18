import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import *

def download_sample_data():
    """샘플 데이터를 다운로드하고 준비합니다."""
    print("샘플 데이터 준비 중...")
    
    # 디렉토리 생성
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    # 임시 샘플 데이터 생성
    for i in range(10):
        # 더미 이미지 생성 (랜덤한 패턴으로)
        image = Image.new('RGB', (256, 256))
        pixels = image.load()
        for x in range(256):
            for y in range(256):
                pixels[x, y] = (x % 256, y % 256, (x * y) % 256)
        
        # 더미 마스크 생성 (간단한 원형 패턴)
        mask = Image.new('L', (256, 256), 0)
        pixels = mask.load()
        center = (128, 128)
        for x in range(256):
            for y in range(256):
                distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
                pixels[x, y] = 255 if distance < 64 else 0
        
        # 이미지 저장
        image_path = os.path.join(IMAGES_DIR, f'image_{i}.png')
        mask_path = os.path.join(MASKS_DIR, f'mask_{i}.png')
        
        image.save(image_path)
        mask.save(mask_path)
    
    print(f"샘플 데이터 생성 완료! ({len(os.listdir(IMAGES_DIR))} 개의 이미지)")

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None):
        """
        Args:
            images_dir (str): 이미지 디렉토리 경로
            masks_dir (str, optional): 마스크 디렉토리 경로
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # 이미지 파일 목록
        self.images = [f for f in os.listdir(images_dir) if f.startswith('image')]
        self.images.sort()  # 순서 보장
        
        # 마스크 파일 목록 (있는 경우)
        if masks_dir:
            self.masks = [f for f in os.listdir(masks_dir) if f.startswith('mask')]
            self.masks.sort()  # 순서 보장
        
        # 변환
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 이미지 로드 및 변환
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 마스크 로드 및 변환 (있는 경우)
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask = Image.open(mask_path).convert('L')  # 그레이스케일로 변환
            mask = self.mask_transform(mask)
            return image, mask
        
        return image

def get_data_loader(batch_size=BATCH_SIZE, shuffle=True):
    """데이터 로더를 생성하고 반환합니다."""
    # 데이터가 없으면 다운로드
    if not os.path.exists(IMAGES_DIR) or len(os.listdir(IMAGES_DIR)) == 0:
        download_sample_data()
    
    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
