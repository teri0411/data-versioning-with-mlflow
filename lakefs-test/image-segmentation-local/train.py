from utils.dir_utils import ensure_directories
from train import ModelTrain

def main():
    """메인 함수"""
    ensure_directories()
    trainer = ModelTrain()
    trainer.train()

if __name__ == "__main__":
    main()
