import mlflow
from config import *

def setup_mlflow():
    """MLflow 설정을 초기화합니다."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def get_experiment_runs():
    """MLflow에서 실험 목록을 가져옵니다."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return []
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id])

def format_run_info(run):
    """실험 정보를 포맷팅합니다."""
    # Git commit hash 가져오기
    git_commit = run.data.tags.get("mlflow.source.git.commit", "N/A")
    if git_commit != "N/A":
        git_commit = git_commit[:8]  # 앞 8자리만 표시
    
    # 파라미터와 메트릭 가져오기
    params = {k: v for k, v in run.data.params.items()}
    metrics = {k: round(float(v), 4) for k, v in run.data.metrics.items()}
    
    return {
        "run_id": run.info.run_id,
        "git_commit": git_commit,
        "params": params,
        "metrics": metrics
    }

def select_run():
    """사용자가 실험을 선택할 수 있게 합니다."""
    runs = get_experiment_runs()
    
    if len(runs) == 0:
        print("저장된 실험이 없습니다.")
        return None
    
    print("사용 가능한 실험 목록:\n")
    for idx, run_info in runs.iterrows():
        run = mlflow.get_run(run_info.run_id)
        info = format_run_info(run)
        
        print(f"{idx + 1}. Run ID: {info['run_id']}")
        print(f"   Git Commit: {info['git_commit']}")
        print("   Parameters:")
        for k, v in info['params'].items():
            print(f"   - {k}: {v}")
        print("   Metrics:")
        for k, v in info['metrics'].items():
            print(f"   - {k}: {v}")
        print()
    
    # 실험 선택
    while True:
        try:
            choice = input("사용할 실험 번호를 선택하세요: ")
            if not choice.strip():  # 빈 입력 처리
                print("실험 선택이 취소되었습니다.")
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(runs):
                run_id = runs.iloc[choice - 1].run_id
                return mlflow.get_run(run_id)
            else:
                print(f"1부터 {len(runs)}까지의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except EOFError:
            print("실험 선택이 취소되었습니다.")
            return None
