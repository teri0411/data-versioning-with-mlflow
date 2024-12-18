# 와인 품질 예측 프로젝트

이 프로젝트는 MLflow와 DVC를 결합하여 효율적인 ML 실험 및 데이터/모델 관리를 구현한 예제입니다:
- MLflow: 실험 메타데이터(파라미터, 메트릭, 아티팩트 경로) 추적
- DVC: 실제 데이터 파일과 모델 파일의 버전 관리

## 데이터/모델 관리 전략

1. **.gitignore 설정**
   ```
   data/          # 데이터 디렉토리
   models/        # 모델 디렉토리
   !data/*.dvc    # DVC 파일은 추적
   !models/*.dvc  # DVC 파일은 추적
   ```

2. **DVC로 관리**
   - `.dvc` 파일만 Git에 저장
   - 실제 파일은 원격 저장소(예: S3)에 저장
   - 학습 스크립트는 DVC에서 데이터만 가져오기
   ```bash
   dvc add data/wine-quality.csv
   dvc add models/model.pt
   ```

3. **MLflow로 추적**
   - 학습 실행마다 다음 정보를 MLflow에 기록:
     - 데이터 경로: `data/wine-quality.csv`의 DVC 경로
     - 모델 경로: `models/model.pt`의 DVC 경로
     - 학습 파라미터
     - 평가 메트릭
     - Git 커밋 정보

## 작업 흐름

1. **데이터 및 모델 준비**
   ```bash
   # DVC에 데이터/모델 추가
   dvc add data/wine-quality.csv
   dvc add models/model.pt
   dvc push
   git add *.dvc
   git commit -m "Add data and model"
   ```

2. **학습 단계**
   ```bash
   python train.py  # 1. DVC에서 데이터 확인
                    # 2. 있으면 DVC에서 가져오기
                    # 3. 없으면 UCI에서 다운로드
                    # 4. 모델 학습
                    # 5. MLflow에 메타데이터 등록
   ```

3. **추론 단계**
   ```bash
   python infer.py  # 1. MLflow에서 최신 실행 선택
                    # 2. MLflow에서 데이터/모델의 DVC 경로 확인
                    # 3. 해당 DVC 경로의 파일들 자동 다운로드
                    # 4. 다운로드된 모델과 데이터로 예측 수행
   ```

## 프로젝트 구조

```
dvc-test/
├── .dvc/               # DVC 설정
├── data/               # 데이터 디렉토리 (DVC로 관리)
│   └── wine-quality.csv    # 와인 품질 데이터셋
├── models/             # 모델 디렉토리 (DVC로 관리)
├── train/             # 학습 모듈
│   ├── base_train.py      # 기본 학습 클래스
│   └── dvc_train.py       # DVC 관련 학습 구현
├── inference/         # 추론 모듈
│   ├── base_inference.py  # 기본 추론 클래스
│   └── mlflow_inference.py # MLflow 관련 추론 구현
├── config.py         # 설정 파일
├── model.py          # 모델 정의
├── train.py          # 학습 스크립트
├── infer.py          # 추론 스크립트
├── utils.py          # 유틸리티 함수
├── setting.py        # 프로젝트 설정 및 환경 변수
└── requirements.txt  # 프로젝트 의존성
```

## 주요 파일

- `train.py`: 
  - 데이터 로딩 및 모델 학습
  - MLflow에 메타데이터 기록
  - DVC를 통한 모델 저장
- `infer.py`: 
  - MLflow에서 실행 정보 조회
  - DVC를 통한 데이터/모델 가져오기
  - 예측 수행
- `setting.py`:
  - AWS 자격 증명과 DVC 설정을 자동화하는 파일입니다.

  1. **AWS 설정 자동화**
     ```python
     # AWS 설정 파일 생성
     config_path = "aws/config"        # AWS 리전 설정
     credentials_path = "aws/credentials"  # AWS 자격 증명
     ```
     - AWS Access Key와 Secret Key를 CLI에서 입력받아 자격 증명 파일 생성
     - AWS 리전은 'ap-northeast-2'로 자동 설정

  2. **DVC 설정 자동화**
     ```python
     # DVC 설정 파일 생성
     dvc_config_path = ".dvc/config"
     ```
     - DVC 원격 저장소: `s3://dataversion-test/dvc`
     - AWS 설정 파일 연동:
       - configpath: AWS 리전 설정 파일 경로
       - credentialpath: AWS 자격 증명 파일 경로
       - profile: default

  3. **실행 방법**
     ```bash
     python setting.py  # AWS 자격 증명 입력 후 설정 파일 자동 생성
     ```

## 의존성

필요한 Python 패키지:
- pandas
- numpy
- torch
- scikit-learn
- mlflow
- dvc

## 사용법

1. **환경 설정**
   ```bash
   pip install -r requirements.txt
   
   # DVC 초기화 및 원격 저장소 설정
   dvc init
   dvc remote add -d storage s3://my-bucket/dvc-store
   ```

2. **학습**
   ```bash
   python train.py [alpha] [l1_ratio]
   ```
   - 선택적 파라미터:
     - `alpha`: Elasticnet 혼합 파라미터 (기본값: 0.5)
     - `l1_ratio`: L1 비율 파라미터 (기본값: 0.5)

3. **추론**
   ```bash
   python infer.py
   ```
   - MLflow에서 최신 실행을 선택
   - DVC를 통해 필요한 데이터/모델 다운로드
   - 예측 수행

## 데이터 형식

와인 품질 데이터셋은 다음 형식을 사용합니다:
- 구분자: 세미콜론 (;)
- 특성: fixed acidity, volatile acidity, citric acid 등
- 타겟: quality (0~10 사이의 점수)

## 참고사항

- Git: 코드와 설정 파일만 관리
- DVC: 데이터와 모델 파일의 버전 관리
- MLflow: 실험 메타데이터 추적 및 데이터/모델 경로 관리
- 학습 지표로 RMSE, MAE, R2 점수 사용

## 문제 해결

### DVC 관련 문제
1. `dvc pull` 실패 시
   - 원격 저장소 설정 확인
   ```bash
   dvc remote list
   dvc remote verify
   ```
   - 인증 정보 확인

2. 데이터/모델 불일치
   - MLflow 실행 정보 확인
   - DVC 캐시 초기화 후 다시 시도
   ```bash
   dvc gc -w
   dvc pull
   ```

```python
DVC_REMOTE = "s3://my-bucket"  # DVC 원격 저장소 URL
DVC_DATA_PATH = "data/wine-quality.csv"  # 데이터 파일 경로
DVC_MODEL_PATH = "models/model.pt"       # 모델 파일 경로

MLFLOW_TRACKING_URI = "http://localhost:5000"  # MLflow 서버 주소
EXPERIMENT_NAME = "wine-quality"               # MLflow 실험 이름

DATA_URL = "https://archive.ics.uci.edu/ml/wine-quality.csv"  # UCI 데이터셋 URL
FEATURES = ["fixed acidity", "volatile acidity", ...]  # 학습에 사용할 특성
TARGET = "quality"                                     # 예측 대상 변수

TRAIN_SIZE = 0.8       # 학습/테스트 데이터 분할 비율
RANDOM_SEED = 42       # 랜덤 시드
LEARNING_RATE = 0.001  # 학습률
BATCH_SIZE = 32        # 배치 크기
EPOCHS = 100          # 학습 에포크
