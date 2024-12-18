import os
from minio import Minio
from config import *

def setup_minio_client():
    """MinIO 클라이언트를 설정하고 반환합니다."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def ensure_bucket_exists(client, bucket_name):
    """버킷이 존재하는지 확인하고, 없으면 생성합니다."""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully!")
        return True
    except Exception as e:
        print(f"Error ensuring bucket exists: {str(e)}")
        return False

def upload_to_minio(client, bucket_name, local_path, object_name):
    """파일을 MinIO에 업로드합니다."""
    try:
        client.fput_object(bucket_name, object_name, local_path)
        return True
    except Exception as e:
        print(f"Error uploading {local_path} to MinIO: {str(e)}")
        return False

def download_from_minio(client, bucket_name, object_name, local_path):
    """MinIO에서 파일을 다운로드합니다."""
    try:
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.fget_object(bucket_name, object_name, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {object_name} from MinIO: {str(e)}")
        return False
