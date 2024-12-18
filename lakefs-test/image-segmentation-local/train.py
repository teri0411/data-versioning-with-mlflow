import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import subprocess

# LakeFS 설정
LAKEFS_ENDPOINT = "http://localhost:8003"
LAKEFS_ACCESS_KEY = "AKIAIOSFOLKFSSAMPLES"
LAKEFS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
REPO_NAME = "image-segmentation-local-repo"

# MLflow 설정
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Image Segmentation"
mlflow.set_experiment(experiment_name)

# LakeFS 리포지토리 생성
configuration = lakefs_client.Configuration()
configuration.host = LAKEFS_ENDPOINT
configuration.username = LAKEFS_ACCESS_KEY
configuration.password = LAKEFS_SECRET_KEY
client = LakeFSClient(configuration)

try:
    # 리포지토리가 이미 존재하는지 확인
    try:
        client.repositories.get_repository(REPO_NAME)
        print(f"Repository {REPO_NAME} already exists")
    except lakefs_client.exceptions.ApiException as e:
        if e.status == 404:  # 리포지토리가 없는 경우에만 생성
            client.repositories.create_repository(models.RepositoryCreation(
                name=REPO_NAME,
                storage_namespace=f"s3://image-segmentation-local-repo",
                default_branch="main"
            ))
            print(f"Created repository: {REPO_NAME}")
        else:
            raise e
except lakefs_client.exceptions.ApiException as e:
    raise e

# SimpleCNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return None

def train_model():
    # 데이터 디렉토리 생성
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Git commit hash 가져오기
    git_commit_hash = get_git_commit_hash()

    # 예시 이미지와 마스크 생성
    num_samples = 100
    print("생성한 이미지 및 마스크:")
    for i in range(num_samples):
        # 가상의 이미지 생성 (실제로는 실제 이미지를 사용)
        image = torch.randn(3, 64, 64)  # RGB 이미지
        # 이미지에 간단한 패턴 추가 (예: 중앙에 원)
        center_x, center_y = 32, 32
        radius = 20
        for x in range(64):
            for y in range(64):
                if (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2:
                    image[:, x, y] = 1.0  # 원 영역을 흰색으로

        # 마스크 생성 (원 영역을 1로 설정)
        mask = torch.zeros(1, 64, 64)
        for x in range(64):
            for y in range(64):
                if (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2:
                    mask[0, x, y] = 1.0

        # 이미지와 마스크를 PNG 파일로 저장
        image_path = os.path.join(images_dir, f"image_{i}.png")
        mask_path = os.path.join(masks_dir, f"mask_{i}.png")
        torchvision.utils.save_image(image, image_path)
        torchvision.utils.save_image(mask, mask_path)
        
        if i < 5:  # 처음 5개 이미지에 대해서만 출력
            print(f"  - {image_path}")
            print(f"  - {mask_path}")

    # 데이터셋 생성
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    class SegmentationDataset(Dataset):
        def __init__(self, images_dir, masks_dir, transform=None):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.transform = transform
            self.images = sorted(os.listdir(images_dir))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = self.images[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name.replace('image', 'mask'))
            
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            return image, mask

    # 데이터셋과 데이터로더 생성
    dataset = SegmentationDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 모델, 손실함수, 옵티마이저 설정
    model = SimpleCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # MLflow 실험 시작
    with mlflow.start_run() as run:
        if git_commit_hash:
            mlflow.set_tag("git_commit_hash", git_commit_hash)
        
        print("\n학습 시작...")
        # 학습 루프
        for epoch in range(10):
            total_loss = 0
            for images, masks in dataloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("loss", avg_loss, step=epoch)

        # 모델 저장
        model_path = "models/simple_cnn_model.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"\n모델 저장 완료: {model_path}")

        # LakeFS 클라이언트 설정
        configuration = lakefs_client.Configuration(
            host=LAKEFS_ENDPOINT
        )
        configuration.username = LAKEFS_ACCESS_KEY
        configuration.password = LAKEFS_SECRET_KEY
        client = LakeFSClient(configuration)

        # LakeFS에 모델 업로드
        branch = "main"
        model_lakefs_path = "models/simple_cnn_model.pt"
        
        print(f"- 모델 파일 업로드: {model_lakefs_path}")
        with open(model_path, 'rb') as f:
            client.objects.upload_object(
                repository=REPO_NAME,
                branch=branch,
                path=model_lakefs_path,
                content=f
            )
            
        # MLflow에 모델 경로와 Git commit hash 기록
        mlflow.log_param("model_path", f"lakefs://{REPO_NAME}/{branch}/{model_lakefs_path}")
        if git_commit_hash:
            mlflow.set_tag("git_commit_hash", git_commit_hash)

        # LakeFS에 데이터 디렉토리 업로드
        print("- 데이터 파일 업로드:")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):  # 이미지 파일만 업로드
                    local_path = os.path.join(root, file)
                    lakefs_path = os.path.relpath(local_path, '.')  # 상대 경로 유지
                    
                    print(f"  - {lakefs_path}")
                    with open(local_path, 'rb') as f:
                        client.objects.upload_object(
                            repository=REPO_NAME,
                            branch=branch,
                            path=lakefs_path,
                            content=f
                        )

        # MLflow에 메타데이터 기록
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_artifact(model_path)
        mlflow.log_param("lakefs_data_path", f"lakefs://{REPO_NAME}/main/data")

if __name__ == "__main__":
    train_model()
