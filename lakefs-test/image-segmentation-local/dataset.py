import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import *

class SegmentationDataset(Dataset):
    """이미지 세그멘테이션을 위한 데이터셋 클래스"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.startswith('image')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('image', 'mask'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def get_data_loader(batch_size=BATCH_SIZE, shuffle=True):
    """데이터 로더를 생성하고 반환합니다."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    ])
    
    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
