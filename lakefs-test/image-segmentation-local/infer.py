from utils.dir_utils import ensure_directories
from inference import ModelInference

def main():
    """메인 함수"""
    ensure_directories()
    inferencer = ModelInference()
    inferencer.infer()

if __name__ == "__main__":
    main()
