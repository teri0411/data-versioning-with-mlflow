import os
import torch
import numpy as np
from PIL import Image
from config import *
from model import create_model

class BaseInference:
    """Base model inference class"""
    
    def __init__(self):
        """Initialize"""
        self.model = create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def load_model(self, model_path):
        """Load the model."""
        print(f"\nLoading model... ({model_path})")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        return self.model
    
    def infer_images(self):
        """Perform inference on all images."""
        results = []
        images_dir = os.path.join(DATA_DIR, "images")
        
        print("\nStarting inference...")
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(".png"):
                continue
                
            image_path = os.path.join(images_dir, image_file)
            output = self.infer_image(image_path)
            results.append((image_file, output))
            print(f"Image processed: {image_file}")
            
        return results
    
    def infer_image(self, image_path):
        """Perform inference on a single image."""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image) / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(image)
            
        # Post-processing
        output = torch.sigmoid(output).cpu().numpy()
        output = (output > 0.5).astype(np.uint8) * 255
        return output
    
    def save_results(self, results):
        """Save inference results."""
        print("\nSaving results...")
        output_dir = os.path.join(DATA_DIR, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        for image_file, output in results:
            output_image = Image.fromarray(output.squeeze())
            output_path = os.path.join(output_dir, image_file.replace("image", "pred"))
            output_image.save(output_path)
            
        print(f"Results have been saved to {output_dir}")
