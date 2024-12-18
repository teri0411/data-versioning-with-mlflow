# Data Versioning Examples

이 저장소는 데이터 버전 관리를 위한 두 가지 접근 방식을 보여주는 예제 프로젝트를 포함하고 있습니다.

## Projects

### 1. DVC Project (Wine Quality Prediction with MLOps)

[dvc-test/](./dvc-test)

DVC(Data Version Control)와 MLflow를 사용한 ML 프로젝트 예제입니다:
- DVC로 데이터와 모델 버전 관리
- MLflow로 실험 추적 및 모델 레지스트리 관리
- 와인 품질 예측 모델을 통한 실제 사용 예시

주요 기능:
- 데이터/모델 버전 관리 (DVC)
- 실험 메타데이터 추적 (MLflow)
- 모델 학습 및 등록 파이프라인
- AWS S3 연동

### 2. LakeFS Project (Image Segmentation with MLOps)

[lakefs-test/](./lakefs-test)

LakeFS를 사용한 데이터 버전 관리와 MLOps 파이프라인 예제입니다:
- LakeFS로 데이터 버전 관리
- MLflow로 실험 추적
- 이미지 세그멘테이션 모델 학습 및 추론

주요 기능:
- Git과 유사한 데이터 버전 관리 (LakeFS)
- 브랜치 기반 실험 관리
- 모델 학습 및 등록 파이프라인
- MinIO 연동

## 시작하기

각 프로젝트의 세부 설명과 사용법은 해당 디렉토리의 README.md를 참조하세요:
- [DVC Project README](./dvc-test/README.md)
- [LakeFS Project README](./lakefs-test/README.md)