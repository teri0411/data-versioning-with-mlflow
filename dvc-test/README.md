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

## 주요 스크립트 설명

### 1. setting.py
- AWS S3 원격 스토리지 설정을 위한 스크립트
- AWS 자격 증명 정보를 입력받아 `aws/` 디렉토리에 설정 파일 생성
- **참고**: 이 스크립트는 S3를 원격 스토리지로 사용할 때만 필요

### 2. train.py
- 와인 품질 예측 모델 학습 스크립트
- 특징:
  - DVC를 통해 S3 저장소에서 데이터를 가져옴 (`dvc.api.get_url` 사용)
  - Git 저장소는 데이터의 메타데이터 추적에 사용
  - 모델을 로컬 경로에 저장 (`dvc/test/data/module/test_model.pt`)
  - MLflow를 통한 실험 관리

- **중요**: AWS 자격 증명 필요
  - 데이터를 S3에서 가져오기 위해 필요
  - `aws/` 디렉토리에 자격 증명 파일이 있어야 함 (setting.py로 생성)

### 3. infer.py
- 학습된 모델을 사용한 추론 스크립트
- MLflow run ID를 입력받아 해당 모델로 추론 수행

## 중요 참고사항

### AWS 설정 관련
데이터는 DVC를 통해 S3 저장소에서 관리됩니다:

1. **데이터 접근**
   - `.dvc/config`에 설정된 S3 저장소 정보 사용
   - AWS 자격 증명 필수 (`aws/` 디렉토리)
   - `dvc.api.get_url`을 통해 데이터 접근

2. **데이터 버전 관리**
   - Git 저장소는 데이터의 메타데이터만 추적
   - 실제 데이터는 S3에 저장
   - `dvc add`, `dvc push` 명령어로 새로운 데이터 버전 관리

## DVC 기본 사용법

### 1. 초기 설정 (새 프로젝트시)
```bash
git init
dvc init
```

### 2. 원격 스토리지 설정
```bash
# S3 설정 예시
dvc remote add -d myremote s3://mybucket/myfolder
dvc remote modify myremote profile myprofile
```

### 3. 데이터 변경 관리
```bash
dvc add {파일명}    # 데이터 추적 시작
dvc push           # 원격 저장소에 데이터 업로드
git add .          # DVC 파일 추적
git commit -m "데이터 업데이트"
```

## 실행 방법

1. (선택사항) AWS S3 설정
```bash
python setting.py
```

2. 모델 학습
```bash
python train.py
```

3. 모델 추론
```bash
python infer.py
``` 
