import hashlib
import mlflow.pyfunc
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
import yaml
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import shutil

MODEL_NAME = 'knuh_v5'
core_dir = "core"  # Core folder path
weight_file_name = "model.pt"  # Weight file name in the core folder
weight_file_path = os.path.join(core_dir, weight_file_name)  # Full path to the weight file


# Variables
temp_dir = "temp_core"  # Temporary directory to hold files excluding weight



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

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_weight_already_registered(model_name, file_hash):
    """Check if a weight file with the same hash is already registered."""
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.tags.get("weight_file_hash") == file_hash:
            return True
    return False

def copy_files_excluding_weight(src_dir, dst_dir, weight_file_name):
    """Copy all files from src_dir to dst_dir excluding the weight file."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)  # Remove existing temporary directory
    os.makedirs(dst_dir)  # Create new temporary directory

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file != weight_file_name:  # Exclude weight file
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, src_dir)
                dst_file = os.path.join(dst_dir, rel_path)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)


# Calculate hash of the weight file
weight_file_hash = calculate_file_hash(weight_file_path)

# Determine code path based on weight file status
if is_weight_already_registered(MODEL_NAME, weight_file_hash):
    print(f"Weight file with hash {weight_file_hash} is already registered. Registering other files only.")
    copy_files_excluding_weight(core_dir, temp_dir, weight_file_name)
    code_path = [temp_dir]
else:
    print(f"Weight file with hash {weight_file_hash} is not registered. Registering entire core folder.")
    code_path = [core_dir]

# Register the model
mlflow.pyfunc.log_model(
    artifact_path="model",  # Specify a unique artifact path
    python_model=ModelWrapper(),
    conda_env=conda_env,
    code_path=code_path,  # Dynamically generated code path
    registered_model_name=MODEL_NAME
)

# Add hash tag to the registered model version
client = MlflowClient()
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    if mv.current_stage == "None":  # Newly registered model
        client.set_model_version_tag(MODEL_NAME, mv.version, "weight_file_hash", weight_file_hash)
        print(f"Registered model version {mv.version} with weight file hash {weight_file_hash}.")
        break

# Cleanup temporary directory
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)