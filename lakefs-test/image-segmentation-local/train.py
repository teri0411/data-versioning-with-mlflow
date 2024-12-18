from utils.dir_utils import ensure_directories
from trainers import ModelTrainer

def main():
    """메인 함수"""
    ensure_directories()
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
