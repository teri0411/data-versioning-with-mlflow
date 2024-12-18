# Image Segmentation with LakeFS and MLflow

이 프로젝트는 이미지 세그멘테이션 모델을 학습하고 추론하는 과정에서 LakeFS를 사용하여 데이터를 버전 관리하고, MLflow로 실험을 추적하는 예제입니다.

## 시스템 아키텍처

### 주요 컴포넌트
- **LakeFS**: 데이터와 모델의 버전 관리
- **MinIO**: S3 호환 오브젝트 스토리지
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
- MinIO Console: http://localhost:9001
- MLflow UI: http://localhost:5000

### 인증 정보
- LakeFS:
  - Access Key ID: AKIAIOSFOLKFSSAMPLES
  - Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
- MinIO:
  - Username: minioadmin
  - Password: minioadmin

## 프로젝트 구조
```
.
├── config.py           # 설정 파일 (MLflow, LakeFS, 학습 파라미터)
├── dataset.py         # 데이터셋 관리 클래스
├── docker-compose.yml # Docker 서비스 설정
├── infer.py           # 모델 추론 스크립트
├── model.py           # 모델 아키텍처 정의
├── requirements.txt   # Python 패키지 의존성
├── train.py          # 모델 학습 스크립트
└── utils.py          # 유틸리티 함수
```

## 모듈 설명

### `config.py`
설정 파일로, 프로젝트의 모든 설정값을 중앙 집중화합니다.

```python
# 주요 설정값
MLFLOW_TRACKING_URI = "http://localhost:5000"  # MLflow 서버 주소
EXPERIMENT_NAME = "image-segmentation"         # MLflow 실험 이름

LAKEFS_ENDPOINT = "http://localhost:8003"      # LakeFS 서버 주소
LAKEFS_ACCESS_KEY = "..."                      # LakeFS 접근 키
LAKEFS_SECRET_KEY = "..."                      # LakeFS 비밀 키
REPO_NAME = "image-segmentation-local-repo"    # LakeFS 저장소 이름
BRANCH_NAME = "main"                           # LakeFS 브랜치 이름

# 모델 학습 파라미터
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10
```

### `dataset.py`
데이터셋 관리를 위한 PyTorch 데이터셋 클래스를 정의합니다.

```python
class SegmentationDataset(Dataset):
    """이미지 세그멘테이션을 위한 커스텀 데이터셋"""
    def __init__(self, images_dir, masks_dir):
        # 이미지와 마스크 데이터 로드
        # 데이터 전처리 및 변환 설정

    def __getitem__(self, idx):
        # 이미지와 마스크 쌍 반환
        # 필요한 전처리 적용

    def __len__(self):
        # 데이터셋 크기 반환
```

### `model.py`
세그멘테이션 모델의 아키텍처를 정의합니다.

```python
class SimpleCNN(nn.Module):
    """간단한 CNN 기반 세그멘테이션 모델"""
    def __init__(self):
        # 컨볼루션 레이어
        # 배치 정규화
        # 활성화 함수

    def forward(self, x):
        # 순전파 연산 정의
        # 특징 추출 및 마스크 생성
```

### `utils.py`
프로젝트 전반에서 사용되는 유틸리티 함수들을 제공합니다.

주요 기능:
- Git commit 해시 조회
- LakeFS 클라이언트 설정
- 파일 업로드/다운로드
- 디렉토리 생성 및 관리

```python
def get_git_commit_hash():
    """현재 Git commit 해시 반환"""

def setup_lakefs_client():
    """LakeFS 클라이언트 초기화 및 설정"""

def upload_to_lakefs(client, local_path, lakefs_path):
    """파일을 LakeFS에 업로드"""

def ensure_directories(*dirs):
    """필요한 디렉토리 생성"""
```

### `train.py`
모델 학습을 관리하는 `Trainer` 클래스를 포함합니다.

주요 기능:
- 데이터 로더 초기화
- 모델 학습 및 검증
- 체크포인트 저장
- MLflow 로깅
- LakeFS 데이터 관리

```python
class Trainer:
    """모델 학습 관리 클래스"""
    def __init__(self):
        # 모델, 옵티마이저, 손실 함수 초기화
        # 데이터 로더 설정
        # LakeFS 클라이언트 설정

    def train_epoch(self):
        """한 에포크 학습 수행"""

    def save_model(self):
        """모델 저장 및 업로드"""
```

### `infer.py`
모델 추론을 관리하는 `Inferencer` 클래스를 포함합니다.

주요 기능:
- MLflow에서 실험 정보 로드
- 모델 가중치 로드
- 이미지 세그멘테이션 수행
- 결과 저장 및 평가

```python
class Inferencer:
    """모델 추론 관리 클래스"""
    def __init__(self):
        # MLflow 실험 정보 로드
        # 모델 초기화
        # 추론 설정

    def load_model(self):
        """저장된 모델 로드"""

    def predict(self, image):
        """이미지 세그멘테이션 수행"""
```

## 실행 프로세스

### 1. 모델 학습 (`train.py`)

학습 스크립트는 다음 작업을 수행합니다:

1. LakeFS 리포지토리 생성 또는 연결
2. 데이터셋 준비
   - `SegmentationDataset` 클래스를 사용하여 데이터 로드
   - 데이터를 LakeFS에 저장
3. 모델 학습
   - SimpleCNN 모델 초기화
   - 학습 과정을 MLflow로 추적
   - 에포크별 손실 값 기록
4. 결과 저장
   - 학습된 모델을 `models/model.pth`에 저장
   - 모델과 데이터를 LakeFS에 업로드

실행 방법:
```bash
python train.py
```

#### 학습 결과 예시
```
학습 시작...
Epoch 1, Loss: 0.5945
Epoch 2, Loss: 0.2044
Epoch 3, Loss: 0.0777
Epoch 4, Loss: 0.0200
Epoch 5, Loss: 0.0098
Epoch 6, Loss: 0.0072
Epoch 7, Loss: 0.0070
Epoch 8, Loss: 0.0068
Epoch 9, Loss: 0.0049
Epoch 10, Loss: 0.0044
```

### 2. 모델 추론 (`infer.py`)

추론 스크립트는 다음 작업을 수행합니다:

1. MLflow에서 실험 정보 로드
   - 최신 실험의 Run ID와 메트릭 확인
   - Git commit 해시 추적
2. 모델 로드 및 추론
   - 저장된 모델 가중치 로드
   - 테스트 이미지에 대한 세그멘테이션 수행
3. 결과 평가
   - 예측된 마스크를 results 디렉토리에 저장
   - 성능 메트릭 계산 및 MLflow에 기록

실행 방법:
```bash
python infer.py
```

#### 추론 결과 예시
```
MLflow에서 실험 정보 가져오기...
Run ID: 319b3643e4714ffc87f1eeb7082d2b88
- Git Commit: 3a902a671b98efca1cef9fd8ba27e9e3f49ad99f
- Loss: 0.0044
- Best Loss: 0.0044
- Final Loss: 0.0044
```

## 데이터 버전 관리

### LakeFS 저장소 구조
```
image-segmentation-local-repo/
├── data/
│   ├── images/        # 입력 이미지
│   └── masks/         # 세그멘테이션 마스크
└── models/
    └── model.pth      # 학습된 모델 가중치
```

### MLflow 실험 추적
- 실험 이름: image-segmentation
- 추적 지표:
  - 에포크별 손실값
  - 최종 손실값
  - Git commit 해시
  - 모델 가중치 경로
  - 데이터 버전 정보

MLflow UI (http://localhost:5000)에서 모든 실험 결과와 지표를 확인할 수 있습니다.

## 주의사항
1. 모델 가중치 로드 시 보안 경고
   - PyTorch의 `torch.load` 함수 사용 시 `weights_only=True` 옵션 권장
   - 향후 버전에서 기본값이 변경될 예정

2. 데이터 버전 관리
   - 모든 데이터 변경사항은 LakeFS에 자동으로 기록됨
   - 실험 재현을 위해 데이터 버전과 Git commit 해시가 함께 저장됨

## 데이터 관리

### Git에서 제외되는 파일들
- `models/`: 학습된 모델 파일
- `mlruns/`: MLflow 실험 데이터
- `data/`: 생성된 학습 데이터
- `results/`: 추론 결과
- `__pycache__/`: Python 캐시 파일

## 문제 해결

1. **LakeFS 접속 오류**:
   - LakeFS UI가 접속되지 않는 경우 서비스 로그 확인:
     ```bash
     docker compose --profile local-lakefs logs lakefs
     ```

2. **MinIO 버킷 오류**:
   - MinIO 버킷 생성 상태 확인:
     ```bash
     docker compose --profile local-lakefs logs minio-setup
     ```

3. **데이터 초기화**:
   - 모든 데이터를 초기화하려면:
     ```bash
     docker compose --profile local-lakefs down -v
     rm -rf data models mlruns results
     ```
