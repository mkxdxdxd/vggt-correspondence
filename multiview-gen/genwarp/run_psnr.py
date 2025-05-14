import torch
import torchvision.transforms as T
from PIL import Image
import math

def calculate_psnr(image_path_1, image_path_2):
    # Load images
    img1 = Image.open(image_path_1).convert("RGB")
    img2 = Image.open(image_path_2).convert("RGB")

    # Define transformation to convert images to tensors
    transform = T.ToTensor()

    transform_resize = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512))  # Resize to (512, 512)
    ])

    # Convert images to tensors
    img1_tensor = transform_resize(img1)
    img2_tensor = transform(img2)

    # import pdb; pdb.set_trace()

    # Ensure images are the same size
    assert img1_tensor.shape == img2_tensor.shape, "Images must have the same dimensions"

    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((img1_tensor - img2_tensor) ** 2)
    
    if mse == 0:
        return float('inf')  # Infinite PSNR if images are identical

    # Calculate PSNR
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))  # assuming pixel values are normalized [0, 1]

    return psnr

# Example usage
image_path_1 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/49449400.png"
image_path_2 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/outputs/10_17_15_38_13_three/syn_mask_yes_norm/frame_0.png"
psnr_value = calculate_psnr(image_path_1, image_path_2)
print(f"PSNR value: {psnr_value} dB")