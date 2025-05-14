import torch
import os
import cv2
import torchvision.transforms as T
from PIL import Image
import numpy as np
import imageio
from torchvision.utils import save_image
from time import gmtime, strftime



def apply_heatmap(tensor):
    # Ensure tensor is in the right format (1, 1, 512, 512)
    assert tensor.ndim == 4 and tensor.shape[1] == 1, "Tensor must have shape (1, 1, H, W)"
    
    # Remove batch dimension and convert to numpy array
    image_np = tensor[0, 0].cpu().numpy()
    
    # Normalize the tensor to range 0-255 for visualization
    image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply heatmap (COLORMAP_AUTUMN) using OpenCV
    heatmap_np = cv2.applyColorMap(image_np, cv2.COLORMAP_AUTUMN)
    
    # Convert back to tensor and add batch dimension
    heatmap_tensor = T.ToTensor()(heatmap_np).unsqueeze(0)
    heatmap_tensor = torch.stack((heatmap_tensor[:,2],heatmap_tensor[:,1],heatmap_tensor[:,0]),dim=1)
    
    return heatmap_tensor


def make_video(frame_list, now, output_folder = "outputs/", folder_name=None): 
    samples = torch.stack(frame_list)
    vid = (
        (samples.permute(0,2,3,1) * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    new_dir = output_folder + f"{now}/{folder_name}"
    os.makedirs(new_dir, exist_ok=True)

    video_path = os.path.join(new_dir, "video.mp4")

    imageio.mimwrite(video_path, vid)

    for i, image in enumerate(samples):
        save_image(image, new_dir + f"/frame_{i}.png")


def create_difference_video(folder_1, folder_2, output_video_path):
    # Get list of image frames
    frames_1 = [f for f in os.listdir(folder_1) if f.endswith('.png')]
    frames_2 = [f for f in os.listdir(folder_2) if f.endswith('.png')]

    # Ensure both folders have the same number of frames
    assert len(frames_1) == len(frames_2), "Both folders must have the same number of frames"

    # Define video writer
    sample_frame_path = os.path.join(folder_1, frames_1[0])
    sample_frame = Image.open(sample_frame_path)
    width, height = sample_frame.size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    # Define transformation to convert image to tensor
    transform = T.Compose([T.ToTensor()])

    diffmap_1 = []
    diffmap_2 = []

    # Loop through frames and calculate the difference
    for frame_1_name, frame_2_name in zip(frames_1, frames_2):
        # Load images and convert to tensors
        frame_1 = Image.open(os.path.join(folder_1, frame_1_name)).convert("RGB")
        frame_2 = Image.open(os.path.join(folder_2, frame_2_name)).convert("RGB")

        frame_1 = transform(frame_1)[None,...]
        frame_2 = transform(frame_2)[None,...]

        whitemask = (frame_1 != 0).float()

        diff_2 = torch.sqrt((frame_2 * whitemask - frame_1)**2).sum(dim=1).unsqueeze(1)
        diff_1 = (frame_2 * whitemask - frame_1).sum(dim=1).unsqueeze(1)

        diff = apply_heatmap(diff_1) * whitemask.to("cpu")
        diff_2 = apply_heatmap(diff_2) * whitemask.to("cpu")

        diffmap_1.append(diff[0])
        diffmap_2.append(diff_2[0])

    now = strftime("%m_%d_%H_%M_%S", gmtime())

    make_video(diffmap_1, now, folder_name="diff_1")
    make_video(diffmap_2, now, folder_name="diff_2")

    # Release the video writer
# Example usage

basepath = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/outputs/"
folder_name = "10_17_15_38_13_three"

# /mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/outputs/
# /mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/outputs/

warp = os.path.join(basepath, folder_name, "warp")
corres = os.path.join(basepath, folder_name, "syn_mask_yes_norm")

create_difference_video(warp, corres, 'difference_video.mp4')