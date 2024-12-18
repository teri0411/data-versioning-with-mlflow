import argparse
import mlflow
from train.mlflow_train import MLflowTrain
from train.lakefs_train import LakeFSTrain
from config import LAKEFS_REPO_NAME, LAKEFS_BRANCH

class ModelRegistrar:
    """모델 등록을 담당하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.mlflow_train = MLflowTrain()
        self.lakefs_train = LakeFSTrain()
    
    def register_model(self, auto_register=True):
        """
        모델을 MLflow Model Registry에 등록합니다.
        
        Args:
            auto_register (bool): 자동으로 모델을 등록할지 여부
        """
        # LakeFS에서 모델 존재 확인
        model_path = f"lakefs://{LAKEFS_REPO_NAME}/{LAKEFS_BRANCH}/models/model.pth"
        if not self.lakefs_train.check_model_exists():
            raise Exception("Model not found in LakeFS")
        
        # MLflow에서 최근 실험 결과 확인
        if auto_register:
            runs = mlflow.search_runs(order_by=["start_time DESC"])
            if len(runs) == 0:
                raise Exception("No MLflow runs found")
            run = runs.iloc[0]
        else:
            # 수동으로 실험 선택
            print("\n=== MLflow 실험 목록 ===")
            runs = mlflow.search_runs(order_by=["start_time DESC"])
            for i, run in runs.iterrows():
                print(f"{i}. Run ID: {run.run_id}")
                print(f"   Start Time: {run.start_time}")
                print(f"   Metrics: {run.metrics}")
                print()
            
            idx = input("등록할 실험 번호를 선택하세요: ")
            run = runs.iloc[int(idx)]
        
        # 모델 메타데이터 등록
        metrics = {k: v for k, v in run.data.metrics.items()}
        self.mlflow_train.register_model(run.run_id, metrics)
        print(f"\n모델이 성공적으로 등록되었습니다.")
        print(f"Run ID: {run.run_id}")
        print(f"Model Path: {model_path}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='모델 등록')
    parser.add_argument('--manual', action='store_true', help='실험을 수동으로 선택합니다.')
    args = parser.parse_args()
    
    registrar = ModelRegistrar()
    registrar.register_model(auto_register=not args.manual)

if __name__ == "__main__":
    main()
