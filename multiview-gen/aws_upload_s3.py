import boto3
import os

def upload_directory(local_directory, bucket_name, s3_folder):
    """
    Uploads all files from a local directory to a specified folder in an S3 bucket.
    
    :param local_directory: Path to the local directory.
    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Destination folder (prefix) in the S3 bucket (e.g., 'realestate_train/').
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            # Construct the relative path to preserve the directory structure
            relative_path = os.path.relpath(local_path, local_directory)
            # Create the S3 object key by joining the destination folder and the relative path
            s3_path = os.path.join(s3_folder, relative_path).replace("\\", "/")
            
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            bucket.upload_file(local_path, s3_path)

if __name__ == "__main__":
    # Set AWS credentials via environment variables (for testing purposes)
    os.environ['AWS_ACCESS_KEY_ID'] = "AKIA57VDL2AFYRXQGFTZ"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
    os.environ['AWS_DEFAULT_REGION'] = "us-east-2"

    bucket_name = 'naver-3d-dataset'
    s3_folder = 'checkpoints/ult'  # Ensure this is the target folder in your S3 bucket
    local_directory = '/media/multiview-gen/checkpoints/ult'
    
    upload_directory(local_directory, bucket_name, s3_folder)