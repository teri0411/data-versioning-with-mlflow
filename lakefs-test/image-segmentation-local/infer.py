from utils.dir_utils import ensure_directories
import argparse
from inference.model_inference import ModelInference

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Image Segmentation Inference')
    parser.add_argument('--manual', action='store_true', help='Manually select the experiment.')
    args = parser.parse_args()
    
    ensure_directories()
    inferencer = ModelInference()
    inferencer.infer(auto_select=not args.manual)

if __name__ == "__main__":
    main()
