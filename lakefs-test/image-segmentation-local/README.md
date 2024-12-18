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
├── docker-compose.yml    # Docker 서비스 설정
├── requirements.txt      # Python 패키지 의존성
├── train.py             # 모델 학습 스크립트
└── infer.py             # 모델 추론 스크립트
```

## 실행 프로세스

### 1. 모델 학습 (`train.py`)

학습 스크립트는 다음 작업을 수행합니다:

1. LakeFS 리포지토리 생성 또는 연결
2. 학습 데이터 생성
   - 랜덤한 이미지와 마스크 생성
   - 데이터를 LakeFS에 저장
3. 모델 학습
   - SimpleCNN 모델 초기화
   - 학습 과정을 MLflow로 추적
4. 결과 저장
   - 학습된 모델을 저장
   - 모델과 데이터를 LakeFS에 업로드

실행 방법:
```bash
python train.py
```

### 2. 모델 추론 (`infer.py`)

추론 스크립트는 다음 작업을 수행합니다:

1. MLflow에서 최신 실험 정보 로드
2. LakeFS에서 모델과 데이터 다운로드
3. 모델을 사용하여 이미지 추론
4. 결과 저장 및 평가
   - 예측된 마스크를 results 디렉토리에 저장
   - 실제 마스크와 비교하여 정확도 계산

실행 방법:
```bash
python infer.py
```

## 데이터 관리

### LakeFS 저장소 구조
```
image-segmentation-local-repo/
├── data/
│   ├── images/          # 입력 이미지
│   └── masks/           # 세그멘테이션 마스크
└── models/              # 학습된 모델 파일
```

### Git에서 제외되는 파일들
- `models/`: 학습된 모델 파일
- `mlruns/`: MLflow 실험 데이터
- `data/`: 생성된 학습 데이터
- `results/`: 추론 결과
- `__pycache__/`: Python 캐시 파일

## 주의사항

1. **데이터 초기화**: 
   - Docker 볼륨을 완전히 초기화하려면: `docker compose --profile local-lakefs down -v`

2. **서비스 의존성**:
   - LakeFS는 MinIO가 시작된 후에 실행됩니다.
   - MinIO 버킷은 minio-setup 서비스에 의해 자동으로 생성됩니다.

3. **스토리지 관리**:
   - MinIO 데이터는 Docker 볼륨 `minio_data`에 저장됩니다.
   - LakeFS 메타데이터는 Docker 볼륨 `lakefs_metadata`에 저장됩니다.

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
