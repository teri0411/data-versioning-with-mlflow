from utils.dir_utils import ensure_directories
import argparse
from inference.model_inference import ModelInferencer

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이미지 세그멘테이션 추론')
    parser.add_argument('--interactive', action='store_true', help='실험을 수동으로 선택합니다.')
    args = parser.parse_args()
    
    ensure_directories()
    inferencer = ModelInferencer()
    inferencer.infer(auto_select=not args.interactive)

if __name__ == "__main__":
    main()
