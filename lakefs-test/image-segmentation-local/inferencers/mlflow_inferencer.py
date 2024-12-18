from utils.mlflow_utils import setup_mlflow, select_run

class MLflowInferencer:
    """MLflow 관련 기능을 처리하는 클래스"""
    
    def __init__(self):
        setup_mlflow()
    
    def select_experiment(self):
        """MLflow에서 실험을 선택합니다."""
        print("MLflow에서 실험 정보 가져오기...\n")
        run = select_run()
        if run is None:
            return None
        
        # Git commit hash 가져오기
        git_commit = run.data.tags.get("mlflow.source.git.commit", "N/A")
        if git_commit != "N/A":
            git_commit = git_commit[:8]  # 앞 8자리만 표시
        
        # 파라미터와 메트릭 가져오기
        params = {k: v for k, v in run.data.params.items()}
        metrics = {k: round(float(v), 4) for k, v in run.data.metrics.items()}
        
        print("\n선택한 실험 정보:")
        print(f"- Run ID: {run.info.run_id}")
        print(f"- Git Commit: {git_commit}")
        print("  Parameters:")
        for k, v in params.items():
            print(f"  - {k}: {v}")
        print("  Metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v}")
        
        return run
