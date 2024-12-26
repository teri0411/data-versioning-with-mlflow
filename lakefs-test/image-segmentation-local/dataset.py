import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import *

def download_sample_data():
    """Download and prepare sample data."""
    print("Preparing sample data...")
    
    # Create directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    # Generate temporary sample data
    for i in range(10):
        # Generate dummy image (with random patterns)
        image = Image.new('RGB', (256, 256))
        pixels = image.load()
        for x in range(256):
            for y in range(256):
                pixels[x, y] = (x % 256, y % 256, (x * y) % 256)
        
        # Generate dummy mask (with simple circular pattern)
        mask = Image.new('L', (256, 256), 0)
        pixels = mask.load()
        center = (128, 128)
        for x in range(256):
            for y in range(256):
                distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
                pixels[x, y] = 255 if distance < 64 else 0
        
        # Save images
        image_path = os.path.join(IMAGES_DIR, f'image_{i}.png')
        mask_path = os.path.join(MASKS_DIR, f'mask_{i}.png')
        
        image.save(image_path)
        mask.save(mask_path)
    
    print(f"Sample data generation complete! ({len(os.listdir(IMAGES_DIR))} images)")

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None):
        """
        Args:
            images_dir (str): Path to images directory
            masks_dir (str, optional): Path to masks directory
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # Image file list
        self.images = [f for f in os.listdir(images_dir) if f.startswith('image')]
        self.images.sort()  # Ensure order
        
        # Mask file list (if exists)
        if masks_dir:
            self.masks = [f for f in os.listdir(masks_dir) if f.startswith('mask')]
            self.masks.sort()  # Ensure order
        
        # Transformation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize all images to 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize masks to the same size
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and transform image
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Load and transform mask (if exists)
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask = Image.open(mask_path).convert('L')  # Convert to grayscale
            mask = self.mask_transform(mask)
            return image, mask
        
        return image

def get_data_loader(batch_size=BATCH_SIZE, shuffle=True):
    """Create and return a data loader."""
    # Download data if it doesn't exist
    if not os.path.exists(IMAGES_DIR) or len(os.listdir(IMAGES_DIR)) == 0:
        download_sample_data()
    
    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
