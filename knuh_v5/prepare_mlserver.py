import os
import mlflow
import json

import pandas as pd

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

model_name = 'knuh_v5'
stage = 'latest'
model_uri = f'models:/{model_name}/{stage}'
mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri, dst_path='./.mlflow_model')

model_settings = {
    'name': model_name,
    'implementation': 'mlserver_mlflow.MLflowRuntime',
    'parameters': {
        'uri': '.mlflow_model'
    }
}
with open('model-settings.json', 'w') as f:
    json.dump(model_settings, f, ensure_ascii=False)

model = mlflow.pyfunc.load_model(
    model_uri='./.mlflow_model'
)
