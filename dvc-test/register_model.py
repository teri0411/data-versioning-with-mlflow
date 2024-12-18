import argparse
import mlflow
from train.mlflow_train import MLflowTrain
from train.dvc_train import DVCTrain
from config import *
from utils import get_dvc_paths

class ModelRegistrar:
    """모델 등록을 담당하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.mlflow_train = MLflowTrain()
        self.dvc_train = DVCTrain()
    
    def register_model(self, auto_register=True):
        """
        모델을 MLflow Model Registry에 등록합니다.
        
        Args:
            auto_register (bool): 자동으로 모델을 등록할지 여부
        """
        # DVC에서 모델 존재 확인
        if not os.path.exists(MODEL_PATH):
            raise Exception("Model not found in local path")
            
        dvc_file = f"{MODEL_PATH}.dvc"
        if not os.path.exists(dvc_file):
            raise Exception("Model is not tracked by DVC")
        
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
        
        # DVC 경로 가져오기
        dvc_paths = get_dvc_paths()
        
        # 모델 메타데이터 등록
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_id=run.run_id):
            mlflow.register_model(
                model_uri=dvc_paths['model_path'],
                name="wine_quality_model",
                tags={"source_run": run.run_id}
            )
        
        print(f"\n모델이 성공적으로 등록되었습니다.")
        print(f"Run ID: {run.run_id}")
        print(f"Model Path: {dvc_paths['model_path']}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='모델 등록')
    parser.add_argument('--manual', action='store_true', help='실험을 수동으로 선택합니다.')
    args = parser.parse_args()
    
    registrar = ModelRegistrar()
    registrar.register_model(auto_register=not args.manual)

if __name__ == "__main__":
    main()
