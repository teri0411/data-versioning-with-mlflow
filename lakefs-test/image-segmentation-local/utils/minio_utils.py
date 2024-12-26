import os
from minio import Minio
from config import *

def setup_minio_client():
    """Configure and return MinIO client."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def ensure_bucket_exists(client, bucket_name):
    """Check if bucket exists, create if it doesn't."""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully!")
        return True
    except Exception as e:
        print(f"Error ensuring bucket exists: {str(e)}")
        return False

def upload_to_minio(client, bucket_name, local_path, object_name):
    """Upload file to MinIO."""
    try:
        client.fput_object(bucket_name, object_name, local_path)
        return True
    except Exception as e:
        print(f"Error uploading {local_path} to MinIO: {str(e)}")
        return False

def download_from_minio(client, bucket_name, object_name, local_path):
    """Download file from MinIO."""
    try:
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.fget_object(bucket_name, object_name, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {object_name} from MinIO: {str(e)}")
        return False
