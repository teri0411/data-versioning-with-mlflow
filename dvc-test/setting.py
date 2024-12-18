import os
import getpass

# 경로 설정 - 절대 경로 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "aws", "config")
credentials_path = os.path.join(BASE_DIR, "aws", "credentials")

os.makedirs(os.path.dirname(config_path), exist_ok=True)
os.makedirs(os.path.dirname(credentials_path), exist_ok=True)

# config 파일 내용
config_content = """[default]
region = ap-northeast-2
"""

# CLI에서 AWS 자격 증명 입력 받기
print("AWS 자격 증명을 입력해주세요:")
aws_access_key = input("AWS Access Key ID: ").strip()
aws_secret_key = getpass.getpass("AWS Secret Access Key: ").strip()

# credentials 파일 내용
credentials_content = f"""[default]
aws_access_key_id={aws_access_key}
aws_secret_access_key={aws_secret_key}
"""

# config 파일 생성
with open(config_path, "w") as config_file:
    config_file.write(config_content)

print(f"Config 파일이 생성되었습니다: {config_path}")

# credentials 파일 생성
with open(credentials_path, "w") as credentials_file:
    credentials_file.write(credentials_content)

print(f"Credentials 파일이 생성되었습니다: {credentials_path}")

# DVC config 파일을 직접 텍스트로 처리
dvc_config_path = os.path.join(BASE_DIR, ".dvc", "config")

# DVC config 파일 내용
dvc_config_content = f"""[core]
    remote = test
['remote "test"']
    url = s3://dataversion-test/dvc
    configpath = {config_path}
    credentialpath = {credentials_path}
    profile = default
"""

# DVC config 파일 생성
with open(dvc_config_path, "w") as dvc_config_file:
    dvc_config_file.write(dvc_config_content)

print(f"DVC config 파일이 생성되었습니다: {dvc_config_path}")