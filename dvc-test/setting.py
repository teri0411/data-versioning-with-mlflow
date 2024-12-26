import os
import getpass

# Path settings - Using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "aws", "config")
credentials_path = os.path.join(BASE_DIR, "aws", "credentials")

os.makedirs(os.path.dirname(config_path), exist_ok=True)
os.makedirs(os.path.dirname(credentials_path), exist_ok=True)

# Config file contents
aws_region = input("AWS Region : ").strip()
config_content = f"""[default]
region = {aws_region}
"""

# Get AWS credentials from CLI
print("Please enter your AWS credentials:")
aws_access_key = input("AWS Access Key ID: ").strip()
aws_secret_key = getpass.getpass("AWS Secret Access Key: ").strip()

# Credentials file contents
credentials_content = f"""[default]
aws_access_key_id={aws_access_key}
aws_secret_access_key={aws_secret_key}
"""

# Create config file
with open(config_path, "w") as config_file:
    config_file.write(config_content)

print(f"Config file has been created: {config_path}")

# Create credentials file
with open(credentials_path, "w") as credentials_file:
    credentials_file.write(credentials_content)

print(f"Credentials file has been created: {credentials_path}")

# Process DVC config file directly as text
dvc_config_path = os.path.join(BASE_DIR, ".dvc", "config")

# DVC config file contents
dvc_config_content = f"""[core]
    remote = test
['remote "test"']
    url = s3://dataversion-test/dvc
    configpath = {config_path}
    credentialpath = {credentials_path}
    profile = default
"""

# Create DVC config file
with open(dvc_config_path, "w") as dvc_config_file:
    dvc_config_file.write(dvc_config_content)

print(f"DVC config file has been created: {dvc_config_path}")