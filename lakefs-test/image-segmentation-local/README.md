# Image Segmentation with LakeFS and MLflow

이 프로젝트는 이미지 세그멘테이션 모델을 학습하고 추론하는 과정에서 LakeFS를 사용하여 데이터를 버전 관리하고, MLflow로 실험을 추적하는 예제입니다.

## 시스템 아키텍처

### 주요 컴포넌트
- **LakeFS**: 데이터와 모델의 버전 관리
- **MLflow**: 실험 추적 및 모델 관리

### 워크플로우
1. LakeFS 리포지토리에서 데이터 관리
2. MLflow로 학습 과정 추적
3. 학습된 모델과 데이터를 LakeFS에 저장
4. 저장된 모델과 데이터를 사용하여 추론 수행

## 설치 및 실행 방법

### 사전 요구사항
- Docker와 Docker Compose가 설치되어 있어야 합니다.
- Python 3.8 이상이 필요합니다.

### 환경 설정

1. Python 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Docker 서비스 시작:
```bash
docker compose --profile local-lakefs up -d
```

### 서비스 접근
- LakeFS UI: http://localhost:8003
- MLflow UI: http://localhost:5000

### 인증 정보
- LakeFS:
  - Access Key ID: AKIAIOSFOLKFSSAMPLES
  - Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

## 프로젝트 구조
```
.
├── config.py                # 설정 파일 (MLflow, LakeFS, 학습 파라미터)
├── docker-compose.yml       # Docker 서비스 설정
├── model.py                # 모델 아키텍처 정의
├── requirements.txt        # Python 패키지 의존성
├── train.py               # 모델 학습 스크립트
├── infer.py              # 모델 추론 스크립트
├── train/                # 학습 관련 클래스들
│   ├── __init__.py
│   ├── base_train.py     # 기본 학습 기능
│   ├── lakefs_train.py   # LakeFS 관련 기능
│   ├── mlflow_train.py   # MLflow 관련 기능
│   └── model_train.py    # 전체 학습 과정 관리
├── inference/            # 추론 관련 클래스들
│   ├── __init__.py
│   ├── base_inference.py # 기본 추론 기능
│   ├── lakefs_inference.py # LakeFS 관련 기능
│   ├── mlflow_inference.py # MLflow 관련 기능
│   └── model_inference.py  # 전체 추론 과정 관리
└── utils/               # 유틸리티 함수들
    ├── __init__.py
    ├── dir_utils.py     # 디렉토리 관련 유틸리티
    ├── lakefs_utils.py  # LakeFS 관련 유틸리티
    └── mlflow_utils.py  # MLflow 관련 유틸리티
```

## 모듈 설명

### `config.py`
설정 파일로, 프로젝트의 모든 설정값을 중앙 집중화합니다.

```python
# MLflow 설정
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Image Segmentation"

# LakeFS 설정
LAKEFS_ENDPOINT = "http://localhost:8003"
LAKEFS_ACCESS_KEY = "..."
LAKEFS_SECRET_KEY = "..."
LAKEFS_REPO_NAME = "image-segmentation-local-repo"
LAKEFS_BRANCH = "main"
LAKEFS_MODEL_PATH = "models"
LAKEFS_DATA_PATH = "data"

# 모델 학습 파라미터
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 256
```

### Train 패키지

#### `base_train.py`
기본 모델 학습 기능을 담당하는 클래스입니다.
```python
class BaseTrain:
    """기본 모델 학습 클래스"""
    def __init__(self):
        # 모델, 옵티마이저, 손실 함수 초기화
        
    def train_epoch(self, epoch):
        # 한 에폭 학습 수행
        # 손실값 계산 및 반환
```

#### `lakefs_train.py`
LakeFS 관련 기능을 처리하는 클래스입니다.
```python
class LakeFSTrain:
    """LakeFS 관련 기능을 처리하는 클래스"""
    def save_model(self, model):
        # 모델 저장 및 LakeFS 업로드
        
    def upload_data(self):
        # 학습 데이터를 LakeFS에 업로드
```

#### `mlflow_train.py`
MLflow 관련 기능을 처리하는 클래스입니다.
```python
class MLflowTrain:
    """MLflow 관련 기능을 처리하는 클래스"""
    def log_params(self):
        # 학습 파라미터 기록
        
    def log_metrics(self, metrics):
        # 메트릭 기록
        
    def log_model_path(self, model_path):
        # 모델 경로 기록
```

#### `model_train.py`
전체 학습 과정을 관리하는 클래스입니다.
```python
class ModelTrain:
    """전체 학습 과정을 관리하는 클래스"""
    def train(self):
        # MLflow 실험 시작
        # 모델 학습 수행
        # 메트릭 기록
        # 모델과 데이터 저장
```

### Inference 패키지

#### `base_inference.py`
기본 모델 추론 기능을 담당하는 클래스입니다.
```python
class BaseInference:
    """기본 모델 추론 클래스"""
    def infer_image(self, image_path):
        # 이미지 로드 및 전처리
        # 추론 수행 및 결과 반환
```

#### `lakefs_inference.py`
LakeFS에서 모델과 데이터를 다운로드하는 기능을 처리하는 클래스입니다.
```python
class LakeFSInference:
    """LakeFS 관련 기능을 처리하는 클래스"""
    def download_model(self, model_path):
        # LakeFS에서 모델 다운로드
        
    def download_data(self, data_path, run_id):
        # LakeFS에서 데이터 다운로드
```

#### `mlflow_inference.py`
MLflow에서 실험을 선택하고 관리하는 기능을 처리하는 클래스입니다.
```python
class MLflowInference:
    """MLflow 관련 기능을 처리하는 클래스"""
    def select_experiment(self):
        # MLflow에서 실험 선택
        # 선택된 실험 정보 반환
```

#### `model_inference.py`
전체 추론 과정을 관리하는 클래스입니다.
```python
class ModelInference:
    """전체 추론 과정을 관리하는 클래스"""
    def infer(self):
        # MLflow에서 실험 선택
        # 모델과 데이터 다운로드
        # 추론 수행
        # 결과 저장
```

## 사용 방법

### 모델 학습
```bash
python train.py
```
- MLflow UI에서 학습 과정과 결과를 확인할 수 있습니다.
- 학습된 모델과 데이터는 LakeFS에 자동으로 저장됩니다.

### 모델 추론
```bash
python infer.py
```
- MLflow UI에서 사용할 실험을 선택합니다.
- 선택한 실험의 모델과 데이터를 사용하여 추론을 수행합니다.
- 결과는 `data/predictions` 디렉토리에 저장됩니다.
