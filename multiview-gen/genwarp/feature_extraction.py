import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import os, json
import numpy as np
import cv2

from vggt.heads.head_act import activate_head
from warping_utils import reprojector
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms


torch.cuda.empty_cache() 
def get_rays(H, W, focals, c2w, batch_size, device):
    """
    Get ray origins and directions from a pinhole camera model in PyTorch.

    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length of the camera.
        c2w (torch.Tensor): Camera-to-world transformation matrix of shape (4, 4) or (3, 4).
        
    Returns:
        rays_o (torch.Tensor): Ray origins of shape (H, W, 3).
        rays_d (torch.Tensor): Ray directions of shape (H, W, 3).
    """
    # Create meshgrid for image coordinates (i, j)

    # short_idx = torch.min(torch.tensor([H,W]))
    # factor = 512 * max(H,W)/min(H,W)

    ray_len = 518
    short_side = torch.min(H,W)

    ray_W = ray_len * W / short_side
    ray_H = ray_len * H / short_side

    margin_W = ray_W / 2 - ray_len / 2
    margin_H = ray_H / 2 - ray_len / 2

    i, j = torch.meshgrid(torch.arange(ray_len, dtype=torch.float32), torch.arange(ray_len, dtype=torch.float32), indexing='xy')

    i = i.to(device)
    j = j.to(device)

    # Compute directions (normalized by focal length)
    focals = focals.reshape(-1,1)

    view_num = focals.shape[0]

    dirs_stack = []

    i = i[None,None,...]
    j = j[None,None,...]

    dirs = torch.stack([(i - ray_W * 0.5 + margin_W) / focals[None,...,None], (j - ray_H * 0.5 + margin_H) / focals[None,...,None], torch.ones_like(i.repeat(batch_size,view_num,1,1))], dim=-1)  # Shape (H, W, 3)

    # Apply camera-to-world rotation matrix to directions
    rays_d = torch.sum(dirs[..., None, :] * c2w[..., None, None, :3, :3], dim=-1)  # Shape (H, W, 3)

    # Broadcast ray origins to match shape (H, W, 3)
    rays_o = c2w[..., None, None, :3, -1].expand(rays_d.shape)  # Shape (H, W, 3)

    return rays_o, rays_d



def depth_unnormalize(normalized_depth):
    depth_min = 0.4285
    depth_max = 2.2866
    t_min = torch.tensor(depth_min, device=normalized_depth.device)
    t_max = torch.tensor(depth_max, device=normalized_depth.device)
    depth = t_min + ((normalized_depth + 1) * (t_max - t_min) / 2.0)
    return depth

def apply_heatmap(tensor):
    # Ensure tensor is in the right format (1, 1, 512, 512)
    assert tensor.ndim == 4 and tensor.shape[1] == 1, "Tensor must have shape (1, 1, H, W)"
    
    # Remove batch dimension and convert to numpy array
    image_np = tensor[0, 0].detach().cpu().numpy()
    
    # Normalize the tensor to range 0-255 for visualization
    image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply heatmap (COLORMAP_AUTUMN) using OpenCV
    heatmap_np = cv2.applyColorMap(image_np, cv2.COLORMAP_MAGMA)
    
    # Convert back to tensor and add batch dimension
    heatmap_tensor = T.ToTensor()(heatmap_np).unsqueeze(0)
    heatmap_tensor = torch.stack((heatmap_tensor[:,2],heatmap_tensor[:,1],heatmap_tensor[:,0]),dim=1)
    
    return heatmap_tensor



def apply_mse_heatmap(tensor):
    # Ensure tensor is in the right format (1, 1, 512, 512)
    assert tensor.ndim == 4 and tensor.shape[1] == 1, "Tensor must have shape (1, 1, H, W)"
    
    # Remove batch dimension and convert to numpy array
    image_np = tensor[0, 0].detach().cpu().numpy()
    
    # Normalize the tensor to range 0-255 for visualization
    image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply heatmap (COLORMAP_AUTUMN) using OpenCV
    heatmap_np = cv2.applyColorMap(image_np, cv2.COLORMAP_VIRIDIS)
    
    # Convert back to tensor and add batch dimension
    heatmap_tensor = T.ToTensor()(heatmap_np).unsqueeze(0)
    heatmap_tensor = torch.stack((heatmap_tensor[:,2],heatmap_tensor[:,1],heatmap_tensor[:,0]),dim=1)
    
    return heatmap_tensor



def get_png_files(directory):
    """
    Returns a list of full file paths for all .png files within the specified directory.

    Args:
        directory (str): The path to the directory to search.
    
    Returns:
        list: A list of file paths ending with .png.
    """
    png_files = []
    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)
        # Check if the entry is a file and ends with .png (case-insensitive)
        if os.path.isfile(full_path) and entry.lower().endswith('.png'):
            png_files.append(full_path)
    return png_files


def resize_feature(feature):
    # input 4 mid level features [4, N, 1369, 2048], 1369 is 37 x 37
    # turn it into [4, N, 518x518, 2048]
    # FeatUp, from 37 to 518
    hi = None
    

def compute_pca(tensor: torch.Tensor, C: int):
    """
    Applies PCA on the last dimension of the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, 37, 37, 1024).
        C (int): The target feature dimension for PCA (C <= 1024).

    Returns:
        tuple: A tuple containing:
            - reduced_tensor (torch.Tensor): Tensor with shape (B, 37, 37, C).
            - mean (torch.Tensor): Mean vector used for centering, shape (1024,).
            - top_components (torch.Tensor): PCA projection matrix of shape (1024, C).
    """
    tensor = tensor.reshape(-1, 37, 37, 2048)
    
    V, H, W, D = tensor.shape  # D should be 1024

    # Reshape to (B*H*W, 1024) so each row is a feature vector
    X = tensor.reshape(-1, D)

    # Compute mean along the feature dimension and center the data
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Compute covariance matrix: shape (1024, 1024)
    # Note: Divide by N - 1 for an unbiased estimate of covariance
    N = X_centered.shape[0]
    cov_matrix = torch.matmul(X_centered.T, X_centered) / (N - 1)

    # Since the covariance matrix is symmetric, we can use eigen-decomposition.
    # torch.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Select the top C eigenvectors based on the eigenvalues.
    # We sort the eigenvalues in descending order and choose the corresponding eigenvectors.
    top_indices = torch.argsort(eigenvalues, descending=True)[:C]
    top_components = eigenvectors[:, top_indices]  # shape: (1024, C)

    # Project the centered data onto the top C eigenvectors
    X_reduced = torch.matmul(X_centered, top_components)  # shape: (B*H*W, C)
    
    # import pdb; pdb.set_trace()

    # Reshape the projected data back into (B, 37, 37, C)
    reduced_tensor = X_reduced.reshape(V, H, W, C)

    return reduced_tensor, mean, top_components

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    scene = "llff_fern"
    image_dir = f"/mnt/data1/minkyung/vggt/examples/{scene}/images"
    #num_images = len([f for f in os.listdir(image_dir) if f.endswith(".png")])

    image_names = get_png_files(image_dir)
        
    images = load_and_preprocess_images(image_names).to(device)
    # import pdb; pdb.set_trace()
    to_pil = ToPILImage()
    
    target_idx = 0
    ref_idx = [1, 2, 3]
    all_idx = [target_idx] + ref_idx
    
    crop_transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize shortest side to 512
        transforms.CenterCrop(518),  # Center crop to 512x512
        transforms.ToTensor()    
    ])

    imgss = []

    for img in images:
        pil_image = to_pil(img)
        imgss.append(crop_transform(pil_image))
        
    images_all = torch.stack(imgss).to(device)
    images = images_all[all_idx]
    images_tgt = images[target_idx]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Prediction for camera_pose only
            predictions = model(images, feature = False)
            
    pose_enc = predictions["pose_enc"] # [1, N, 9]
    extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:]) 
    extrinsic_tgt, intrinsic_tgt = extrinsic_all[:, target_idx:target_idx+1,], intrinsic_all[:, target_idx:target_idx+1,]
    
    images_ref = images_all[ref_idx]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            mid_level_feature, predictions = model(images_ref, feature = True)

    # mid_level_feature, [4, N, 1369, 2048], each 4 from 4, 11, 17, 23
    end_level_depth = predictions['depth_feature'] # [N, 128, 518, 518]
    end_level_pts3d = predictions['world_points_feature'] # [N, 128, 518, 518]
    pose_enc = predictions["pose_enc"] # [1, N, 9]
    pointmaps = predictions["world_points"]
    extrinsic_ref, intrinsic_ref = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    extrinsic, intrinsic = torch.cat([extrinsic_tgt, extrinsic_ref], dim=1), torch.cat([intrinsic_tgt, intrinsic_ref], dim=1)
    
    target_idx = 0
    N = len(ref_idx)
    ref_idx = [i for i in range(1, N+1)]
    
    # render ref image in target view
    batch_size, num_view, _, _ = extrinsic.shape

    w2c = torch.cat((extrinsic, torch.tensor([0,0,0,1]).to(device)[None,None,None,...].repeat(batch_size, num_view, 1, 1)), dim=-2)
    extrinsic = torch.linalg.inv(w2c)
    extrinsic = torch.matmul(w2c[:,target_idx].unsqueeze(1), extrinsic)
    pointmaps = torch.matmul(w2c[:,target_idx][:,None,None,None,:3,:3], pointmaps[...,None]).squeeze(-1) + w2c[:,target_idx][:,None,None,None,:3,3]
                            
    focal_list = []

    for i in intrinsic:
        min_val, idx = i[0,:2,2].min(dim=-1)
        focal_list.append(i[:,int(idx),int(idx)] * (259 / min_val))

    focal = torch.stack(focal_list)[...,None]
                                    
    ref_camera=dict(pose=extrinsic[:,ref_idx].float(), 
        focals=focal[:,ref_idx], 
        orig_img_size=torch.tensor([518, 518]).to(device))

    tgt_camera=dict(pose=extrinsic[:,target_idx].float(),
        focals=focal[:,target_idx],
        orig_img_size=torch.tensor([518, 518]).to(device))

    camera_info = dict(ref=ref_camera, tgt=tgt_camera)
    pts_stacked = pointmaps.float().reshape(1, -1, 3).to(device)

    target_pose = camera_info["tgt"]["pose"]
    fovy = camera_info["tgt"]["focals"]

    pts_rgb = images_ref.unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 3)  #[1, -1, 3]
    pts_depth_feat = end_level_depth.unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 128)
    pts_point_feat = end_level_pts3d.unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 128)
    mid_level_resized = resize_feature(mid_level_feature) # FeatUp, [4, N, 518x518, 2048], N = # of src_imgs
    
    warped_image, _ = reprojector(pts_stacked, pts_rgb, target_pose, device, fovy, img_size=518) #[1, 518, 518, 3]
    warped_depth_feat, _ = reprojector(pts_stacked, pts_depth_feat, target_pose, device, fovy, img_size=518) 
    warped_pts_feat, _ = reprojector(pts_stacked, pts_point_feat, target_pose, device, fovy, img_size=518)

    warped_image = warped_image.permute(0,3,1,2)
    warped_depth_feat = warped_depth_feat.permute(0,3,1,2)
    warped_pts_feat = warped_pts_feat.permute(0,3,1,2)
    image_diff = apply_mse_heatmap(torch.sqrt(((warped_image - images_tgt) ** 2).sum(dim=1).unsqueeze(1))) # [1, 3, H, W]
    
    reduced_pts = F.interpolate(pointmaps[0].permute(0,3,1,2), size=37, mode='bilinear', align_corners=False)
    reduced_pts_stacked = reduced_pts.permute(0,2,3,1).reshape(1, -1, 3)
    dim = 3
    
    featstack = []
    
    # import pdb; pdb.set_trace()
    
    featup_upsampler = torch.hub.load(
        "mhamilton723/FeatUp", 
        "dinov2", 
        use_norm=False,       # don’t add the ChannelNorm layer
        ).to(device)

    import pdb; pdb.set_trace()
    
    for f in mid_level_feature:
        reduced_tensor, mean, top_components = compute_pca(f, dim)
        featstack.append(reduced_tensor)
        
    featstack = torch.stack(featstack, dim=0).reshape(4,3,37,37,dim)  # [4, N, 37, 37, 3]
    
    
    featstack_vis = featstack
    
    for i, feat in enumerate(featstack):
        min = torch.quantile(feat.reshape(-1,3), 0.10)
        max = torch.quantile(feat.reshape(-1,3), 0.80)
        
        # if i == 0 or i== 1:
        #     min = 0.0
        
        feat = (feat - min) / (max - min)
        featstack[i] = feat
    
    warps = []
    
    for i, feat in enumerate(featstack):        
        warped_feat, _ = reprojector(reduced_pts_stacked, feat[None,...].reshape(1,-1,dim), target_pose, device, fovy * (37 / 518), img_size=37)
        min = torch.quantile(warped_feat.reshape(-1,3), 0.10)
        max = torch.quantile(warped_feat.reshape(-1,3), 0.80)
        
        warped_feat = (warped_feat - min) / (max - min)
        warps.append(warped_feat)
        
        featstack_vis[i] = (featstack_vis[i] - min) / (max - min)
        
    warped_features = torch.cat(warps, dim=0)
    
    save_image(warped_features.permute(0,3,1,2),'vwarr.png')
    
    per_view_vis = featstack_vis.permute(1,0,4,2,3)
        
    save_image(per_view_vis[0], "view_0.png")
    save_image(per_view_vis[1], "view_1.png")
    save_image(per_view_vis[2], "view_2.png")
        
    import pdb; pdb.set_trace()
    
    
    reduced_pts_stacked
    
    save_image(warped_image, "x_img_warped.png")
    save_image(images_tgt, "x_img_gt.png")
    save_image(image_diff, "x_img_diff.png")

