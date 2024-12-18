# Data Versioning with DVC

이 프로젝트는 DVC(Data Version Control)를 사용한 데이터 버전 관리 테스트 프로젝트입니다.

## 프로젝트 구조
```
dvc-test/
├── .dvc/           # DVC 설정 디렉토리
├── aws/            # AWS 자격 증명 파일 (선택사항)
├── dvc/            # 데이터 디렉토리
├── setting.py      # AWS 설정 스크립트
├── train.py        # 모델 학습 스크립트
├── infer.py        # 모델 추론 스크립트
└── requirements.txt # 의존성 파일
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## AWS S3 설정 방법

### 1. AWS 자격 증명 설정
```bash
python setting.py
```
이 스크립트는 다음 작업을 수행합니다:
- AWS Access Key ID와 Secret Access Key 입력 받기
- `aws/config`와 `aws/credentials` 파일 생성
- `.dvc/config` 파일 자동 생성 및 설정

### 2. 생성되는 설정 파일
- **aws/config**: AWS 리전 설정
  ```ini
  [default]
  region = ap-northeast-2
  ```

- **aws/credentials**: AWS 자격 증명
  ```ini
  [default]
  aws_access_key_id = YOUR_ACCESS_KEY
  aws_secret_access_key = YOUR_SECRET_KEY
  ```

- **.dvc/config**: DVC 설정
  ```ini
  [core]
      remote = test
  ['remote "test"']
      url = s3://dataversion-test/dvc
      configpath = /path/to/aws/config
      credentialpath = /path/to/aws/credentials
  ```

## 주요 스크립트 설명

### 1. setting.py
- AWS S3 원격 스토리지 설정을 위한 스크립트
- AWS 자격 증명 정보를 입력받아 `aws/` 디렉토리에 설정 파일 생성
- DVC config 파일도 자동으로 생성하고 설정
- **참고**: S3를 원격 스토리지로 사용할 때 반드시 실행 필요

### 2. train.py
- 와인 품질 예측 모델 학습 스크립트
- 특징:
  - DVC를 통해 S3 저장소에서 데이터를 가져옴 (`dvc.api.get_url` 사용)
  - Git 저장소는 데이터의 메타데이터 추적에 사용
  - 모델을 로컬 경로에 저장 (`dvc/test/data/module/test_model.pt`)
  - MLflow를 통한 실험 관리

### 3. infer.py
- 학습된 모델을 사용한 추론 스크립트
- MLflow run ID를 입력받아 해당 모델로 추론 수행

## DVC 기본 사용법

### 1. 데이터 가져오기
```bash
dvc pull  # S3에서 데이터 다운로드
```

### 2. 데이터 변경 관리
```bash
dvc add {파일명}    # 데이터 추적 시작
dvc push           # 원격 저장소에 데이터 업로드
git add .          # DVC 파일 추적
git commit -m "데이터 업데이트"
```

## 실행 방법

1. AWS S3 설정 (필수)
```bash
python setting.py  # AWS 자격 증명 입력
dvc pull          # S3에서 데이터 다운로드
```

2. 모델 학습
```bash
python train.py
```

3. 모델 추론
```bash
python infer.py
```

## 문제 해결

### dvc pull 실패 시
1. AWS 자격 증명이 올바르게 설정되었는지 확인
   ```bash
   python setting.py  # AWS 자격 증명 재설정
   ```
2. .dvc/config 파일에서 다음 설정 확인
   - configpath와 credentialpath가 절대 경로인지 확인
