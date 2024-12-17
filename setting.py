import os
import getpass

# 경로 설정
config_path = "./aws/config"
credentials_path = "./aws/credentials"

os.makedirs(os.path.dirname(config_path), exist_ok=True)
os.makedirs(os.path.dirname(credentials_path), exist_ok=True)

# config 파일 내용
config_content = """[corp-prod]
region = ap-northeast-2
"""

# CLI에서 AWS 자격 증명 입력 받기
print("AWS 자격 증명을 입력해주세요:")
aws_access_key = input("AWS Access Key ID: ").strip()
aws_secret_key = getpass.getpass("AWS Secret Access Key: ").strip()

# credentials 파일 내용
credentials_content = f"""[corp-prod]
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
dvc_config_path = "./.dvc/config"

# 설정할 경로들
config_path = "./aws/config"
credentials_path = "./aws/credentials"

# 파일 내용 읽기
with open(dvc_config_path, 'r') as file:
    lines = file.readlines()

# 새로운 설정을 추가할 위치 찾기
new_lines = []
in_test_section = False
config_added = False
credential_added = False

for line in lines:
    # test 섹션 시작 확인
    if "['remote \"test\"']" in line:
        in_test_section = True
        new_lines.append(line)
        continue
    
    # 다음 섹션이 시작되면 test 섹션 종료
    if in_test_section and line.strip().startswith('['):
        if not config_added:
            new_lines.append(f"    configpath = {config_path}\n")
        if not credential_added:
            new_lines.append(f"    credentialpath = {credentials_path}\n")
        in_test_section = False
    
    # test 섹션 내에서 처리
    if in_test_section:
        # profile 라인은 건너뛰기
        if 'profile =' in line:
            continue
        # configpath 처리
        if 'configpath =' in line:
            new_lines.append(f"    configpath = {config_path}\n")
            config_added = True
            continue
        # credentialpath 처리
        if 'credentialpath =' in line:
            new_lines.append(f"    credentialpath = {credentials_path}\n")
            credential_added = True
            continue
    
    new_lines.append(line)

# 파일 끝까지 갔는데 아직 test 섹션이면
if in_test_section:
    if not config_added:
        new_lines.append(f"    configpath = {config_path}\n")
    if not credential_added:
        new_lines.append(f"    credentialpath = {credentials_path}\n")

# 파일에 저장
with open(dvc_config_path, 'w') as file:
    file.writelines(new_lines)