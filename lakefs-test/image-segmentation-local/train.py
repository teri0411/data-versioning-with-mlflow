from utils.dir_utils import ensure_directories
from train import ModelTrain
from config import LAKEFS_REPO_NAME, LAKEFS_BRANCH

def main():
    """메인 함수"""
    ensure_directories()
    trainer = ModelTrain()
    model = trainer.train()  # model.pth 생성
    
    # Git 커밋 및 푸시를 위한 안내 메시지
    print("\n=== 다음 단계 ===")
    print("1. Git에 변경사항을 커밋하세요:")
    print("   git add models/model.pth")
    print("   git commit -m 'Add trained model'")
    print("   git push")
    print("\n2. LakeFS에 모델을 업로드하세요:")
    print(f"   lakectl upload lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth models/model.pth")
    print(f"   lakectl commit lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH} -m 'Add trained model'")
    
    return model

if __name__ == "__main__":
    main()
