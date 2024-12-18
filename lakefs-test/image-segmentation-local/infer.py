from utils.dir_utils import ensure_directories
from inferencers import ModelInferencer

def main():
    """메인 함수"""
    ensure_directories()
    inferencer = ModelInferencer()
    inferencer.infer()

if __name__ == "__main__":
    main()
