import os
import torch
import mlflow
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from train import SimpleCNN
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# LakeFS 설정
LAKEFS_ENDPOINT = "http://localhost:8003"
LAKEFS_ACCESS_KEY = "AKIAIOSFOLKFSSAMPLES"
LAKEFS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
REPO_NAME = "image-segmentation-local-repo"

# MLflow 설정
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Image Segmentation"

def download_file_from_lakefs(client, repository, branch, path, local_path):
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 파일 다운로드
    try:
        response = client.objects.get_object(
            repository=repository,
            ref=branch,
            path=path
        )
        with open(local_path, 'wb') as f:
            f.write(response.read())
        return True
    except Exception as e:
        print(f"Error downloading {path}: {str(e)}")
        return False

def load_model_and_infer():
    print("MLflow에서 최신 실험 가져오기...")
    # MLflow에서 최신 실험 가져오기
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    latest_run = runs.iloc[0]  # 가장 최근 실행 선택

    # LakeFS 클라이언트 설정
    configuration = lakefs_client.Configuration()
    configuration.host = LAKEFS_ENDPOINT
    configuration.username = LAKEFS_ACCESS_KEY
    configuration.password = LAKEFS_SECRET_KEY
    client = LakeFSClient(configuration)

    print("\nLakeFS에서 파일 다운로드 중...")
    # 모델 파일 다운로드
    model_path = "models/simple_cnn_model.pt"
    local_model_path = model_path
    print(f"- 모델 파일: {model_path}")
    download_file_from_lakefs(client, REPO_NAME, "main", model_path, local_model_path)

    # 데이터 디렉토리 다운로드
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # LakeFS에서 이미지와 마스크 파일 리스트 가져오기
    print("- 학습 데이터:")
    try:
        # 이미지와 마스크 파일 다운로드
        for prefix in ["data/images/", "data/masks/"]:
            objects = client.objects.list_objects(
                repository=REPO_NAME,
                ref="main",
                prefix=prefix
            )
            
            for obj in objects.results:
                if obj.path.endswith(('.jpg', '.png')):  # 이미지 파일만 다운로드
                    local_path = obj.path
                    print(f"  - {local_path}")
                    download_file_from_lakefs(client, REPO_NAME, "main", obj.path, local_path)
    except Exception as e:
        print(f"Error listing objects: {str(e)}")
        return

    # 데이터셋 생성
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 테스트용 이미지 로드
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    test_images = sorted(os.listdir(images_dir))[:5]  # 처음 5개 이미지만 테스트

    # 모델 로드
    print("\n추론 시작...")
    print("- 모델 로드")
    model = SimpleCNN()
    model.load_state_dict(torch.load(local_model_path))
    model.eval()

    # 추론 실행
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("- 이미지 추론 및 결과 저장:")
    with torch.no_grad():
        for img_name in test_images:
            # 이미지 로드 및 전처리
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, img_name.replace('image', 'mask'))
            
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
            mask_tensor = transform(mask)

            # 추론
            output = model(image_tensor)
            
            # 결과 저장
            result_path = os.path.join(results_dir, f"pred_{img_name}")
            torchvision.utils.save_image(output, result_path)
            
            # 원본 이미지도 저장
            original_path = os.path.join(results_dir, f"original_{img_name}")
            torchvision.utils.save_image(image_tensor, original_path)
            
            # 실제 마스크도 저장
            mask_save_path = os.path.join(results_dir, f"true_{img_name}")
            torchvision.utils.save_image(mask_tensor, mask_save_path)
            
            print(f"\n이미지: {img_name}")
            print("  입력 이미지:", img_path)
            print("  실제 마스크:", mask_path)
            print("  예측 마스크:", result_path)
            
            # 예측값 출력
            print("\n  예측값 샘플 (5x5):")
            print(output[0, 0, :5, :5])  # 첫 번째 채널의 5x5 영역 출력
            print("\n  실제값 샘플 (5x5):")
            print(mask_tensor[0, :5, :5])  # 첫 번째 채널의 5x5 영역 출력
            
            # 정확도 계산
            pred_binary = (output > 0.5).float()
            accuracy = (pred_binary == mask_tensor).float().mean().item()
            print(f"\n  정확도: {accuracy:.4f}")
            print("-" * 80)

if __name__ == "__main__":
    load_model_and_infer()
