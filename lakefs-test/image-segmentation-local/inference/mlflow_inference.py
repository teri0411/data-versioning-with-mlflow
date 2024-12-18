import mlflow
from config import *

class MLflowInference:
    """MLflow 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        """초기화"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def select_experiment(self, auto_select=True):
        """실험을 선택합니다."""
        print("\nMLflow 실험 목록:")
        runs = mlflow.search_runs(order_by=["start_time DESC"])
        
        if len(runs) == 0:
            print("실험이 없습니다.")
            return None
            
        # 실험 목록 출력
        for idx, (_, run) in enumerate(runs.iterrows()):
            run_details = mlflow.get_run(run.run_id)
            print(f"\n{idx + 1}. Run ID: {run.run_id}")
            print(f"   Git Commit: {run_details.data.tags.get('mlflow.source.git.commit', 'N/A')}")
            print(f"   시작 시간: {run.start_time}")
            
            # Parameters
            params = run_details.data.params
            if params:
                print("   Parameters:")
                for param_name, value in params.items():
                    print(f"   - {param_name}: {value}")
            
            # Metrics
            metrics = run_details.data.metrics
            if metrics:
                print("   Metrics:")
                for metric_name, value in metrics.items():
                    print(f"   - {metric_name}: {value:.4f}")
        
        # 실험 선택
        if auto_select:
            selected_run = mlflow.get_run(runs.iloc[0].run_id)
            print(f"\n최근 실험 선택:")
            print(f"- Run ID: {selected_run.info.run_id}")
            print(f"- Git Commit: {selected_run.data.tags.get('mlflow.source.git.commit', 'N/A')}")
            print(f"- 시작 시간: {selected_run.info.start_time}")
            return selected_run
            
        while True:
            try:
                choice = input("\n실험 번호를 선택하세요 (기본값: 1): ").strip()
                if not choice:
                    choice = "1"
                choice = int(choice)
                if 1 <= choice <= len(runs):
                    selected_run = mlflow.get_run(runs.iloc[choice-1].run_id)
                    print(f"\n선택된 실험:")
                    print(f"- Run ID: {selected_run.info.run_id}")
                    print(f"- Git Commit: {selected_run.data.tags.get('mlflow.source.git.commit', 'N/A')}")
                    print(f"- 시작 시간: {selected_run.info.start_time}")
                    return selected_run
                else:
                    print("올바른 번호를 선택해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")
