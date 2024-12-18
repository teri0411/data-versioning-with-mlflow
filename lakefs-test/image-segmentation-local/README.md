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

## 주요 클래스 및 기능

### ModelInference
모델을 로드하고 이미지 세그멘테이션을 수행하는 기본 클래스입니다.
```python
class ModelInference:
    """이미지 세그멘테이션 모델 추론 클래스"""
    def infer(self, model_path):
        """모델을 로드하고 추론을 수행합니다."""
```

### MLflowInference
MLflow 실험을 관리하고 모델을 선택하는 기능을 처리하는 클래스입니다.
```python
class MLflowInference:
    """MLflow 관련 기능을 처리하는 클래스"""
    def select_experiment(self, auto_select=True):
        """실험을 선택합니다.
        
        Args:
            auto_select (bool): True이면 자동으로 최근 실험을 선택하고,
                              False이면 사용자가 실험을 선택할 수 있습니다.
        """
```

### LakeFSInference
LakeFS에서 모델과 데이터를 다운로드하는 기능을 처리하는 클래스입니다.
```python
class LakeFSInference:
    """LakeFS 관련 기능을 처리하는 클래스"""
    def download_model(self, model_path):
        """LakeFS에서 모델을 다운로드합니다."""
    
    def download_data(self, run_id):
        """LakeFS에서 데이터를 다운로드합니다."""
```

## 추론 결과
추론 결과는 `data/predictions` 디렉토리에 저장됩니다. 각 이미지에 대한 세그멘테이션 마스크가 생성됩니다.
