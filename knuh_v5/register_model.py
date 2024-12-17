import mlflow
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
import pickle
import yaml
import numpy as np
import subprocess

import pandas as pd
print(f"pandas버전: {pd.__version__}")

DEVICE = torch.device("cpu")

MODEL_NAME = 'knuh_v5'

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
# 프로젝트명 (woorim, smartfactory...)
mlflow.set_experiment(MODEL_NAME)

artifacts = {
    "model": "model"
}

# 학습 환경, 서버 환경 맞추기 위해 conda_env 정의
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=sys.version_info.major,
                                                  minor=sys.version_info.minor,
                                                  micro=sys.version_info.micro)
with open('requirements.txt') as f:
    pip_deps = f.read().splitlines()

conda_env = {
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': pip_deps,
        },
    ],
}
with open('./conda.yaml', 'w') as file:
    yaml.dump(conda_env, file)


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        print('init', flush=True)
    def load_context(self, context):
        print('load model', flush=True)
        
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


mlflow.pyfunc.log_model(artifact_path="",
                        python_model=ModelWrapper(),
                        conda_env=conda_env,
                        #artifacts=artifacts,
                        code_path=["core"],
                        registered_model_name=MODEL_NAME)
