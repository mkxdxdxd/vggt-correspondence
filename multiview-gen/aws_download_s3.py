import boto3
import os

def download_s3_folder(bucket_name, s3_folder, local_dir, aws_access_key_id, aws_secret_access_key, region_name):
    """
    Download the contents of an S3 folder to a local directory,
    skipping files that already exist locally with the same size.

    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Folder path in S3 (e.g., 'my-folder/' with trailing slash).
    :param local_dir: Local directory to download files to.
    :param aws_access_key_id: AWS access key.
    :param aws_secret_access_key: AWS secret key.
    :param region_name: AWS region.
    """
    # Create a boto3 session with the provided credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    
    for obj in bucket.objects.filter(Prefix=s3_folder):
        # Remove the folder prefix from the object key to construct the local path.
        local_file_path = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        
        # Create local directory if it doesn't exist.
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Skip if the key is a directory.
        if obj.key.endswith('/'):
            continue
        
        # Check if the file already exists.
        if os.path.exists(local_file_path):
            local_file_size = os.path.getsize(local_file_path)
            if local_file_size == obj.size:
                print(f"Skipping {obj.key}, already exists and is up to date.")
                continue
            else:
                print(f"File {obj.key} exists but size differs, re-downloading.")
        
        print(f"Downloading {obj.key} to {local_file_path}")
        bucket.download_file(obj.key, local_file_path)
        
if __name__ == "__main__":
    bucket_name = 'naver-3d-dataset'
    s3_folder = 'checkpoints/normal_co_train/normal'  # Ensure it ends with a slash if needed.
    local_dir = '/media/multiview-gen/checkpoints/normal_cotrain'

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    aws_access_key_id = "AKIA57VDL2AFYRXQGFTZ"
    aws_secret_access_key = "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"
    region_name = "us-east-2"
    
    download_s3_folder(bucket_name, s3_folder, local_dir, aws_access_key_id, aws_secret_access_key, region_name)
