import mlflow
import os
import sys
import yaml
import subprocess
import os
import shutil
import dvc.api

# 모델을 위한 MLflow 환경 설정
MODEL_NAME = 'knuh_v5'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment(MODEL_NAME)

artifacts = {
    "model": "model"
}

# 학습 환경, 서버 환경 맞추기 위해 conda_env 정의
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=sys.version_info.major,
                                                  minor=sys.version_info.minor,
                                                  micro=sys.version_info.micro)
# with open('requirements.txt') as f:
#     pip_deps = f.read().splitlines()

# conda_env = {
#     'dependencies': [
#         'python={}'.format(PYTHON_VERSION),
#         'pip',
#         {
#             'pip': pip_deps,
#         },
#     ],
# }
# with open('./conda.yaml', 'w') as file:
#     yaml.dump(conda_env, file)

# DVC에서 `core` 폴더 전체를 로컬로 다운로드하는 함수
def load_code_from_dvc(path, repo, version="master"):
    # DVC API를 사용하여 `core` 폴더 URL 가져오기
    code_url = dvc.api.get_url(path=path, repo=repo, rev=version)
    print(code_url)
    # `core` 폴더를 로컬 디렉토리로 다운로드
    local_code_path = "core"  # 로컬로 다운로드할 `core` 폴더 경로
    if not os.path.exists(local_code_path):
        os.makedirs(local_code_path)  # 로컬 폴더 생성
    
    shutil.copytree(code_url, local_code_path)  # DVC에서 `core` 폴더를 로컬로 복사
    
    return local_code_path

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        print('init', flush=True)
    def load_context(self, context):
        print('load model', flush=True)
        # DVC에서 core 폴더를 로컬로 다운로드 후 경로 반환

        
    def predict(self, context, model_input):
        print('predict', flush=True)
        from core.covsf import covsf
        
        inputs = dict([(input.name, [*input.data])
                      for input in model_input.inputs])
        data = inputs['data'][0]
        df = pd.read_json(data)
        CovSF = covsf()
        res = CovSF.run(df)
        print("res")
        print(res)

        return res

repo = "https://www.simplatform.com/gitlab2/teri0411/data-versioning.git"
path = "dvc/knuh_v5/data/module/core"
core_code_path = load_code_from_dvc(path=path, repo=repo)

# MLflow로 모델 로깅
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=ModelWrapper(),
    conda_env=None,
    code_paths=[core_code_path],  # 로컬의 core 폴더 경로를 전달
    registered_model_name=MODEL_NAME
)
