# from huggingface_hub import hf_hub_download

# # Define model repo and filename
# repo_id = "prs-eth/marigold-normals-v0-1"
# filename = "unet"  # Adjust based on actual file name

# # Download the checkpoint file
# file_path = hf_hub_download(repo_id=repo_id, filename=filename)

# print(f"Downloaded checkpoint to: {file_path}")

from huggingface_hub import snapshot_download

# Repository ID from Hugging Face
repo_id = "prs-eth/marigold-normals-v0-1"

# Target directory where files will be downloaded
download_path = "./marigold_normals_v0_1"

# Download only the "unet" folder
snapshot_download(repo_id, local_dir=download_path, allow_patterns=["unet/*"])

print(f"Unet folder downloaded to: {download_path}/unet")