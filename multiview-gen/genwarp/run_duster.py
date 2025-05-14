import argparse
import os
import glob
import torch
import numpy as np
import imageio
import cv2
import math

from PIL import Image
from time import gmtime, strftime
from tqdm import tqdm
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import webdataset as wds
import json

from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from genwarp import GenWarp
# from genwarp.ops import camera_lookat

import  open3d as o3d
from torchvision.transforms import v2

import torchvision.transforms as transforms

import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from genwarp.utils import (
    reprojector,
    ndc_rasterizer,
    get_rays,
    pose_to_ray,
    one_to_one_rasterizer,
    mesh_rendering,
    features_to_world_space_mesh,
    torch_to_o3d_mesh,
    torch_to_o3d_cuda_mesh,
    numpy_to_o3d_cuda_mesh,
    compute_plucker_embed
)


from training_utils import get_embedder
from train_model import prepare_duster_embedding, embedding_prep
from train_marigold import mari_embedding_prep

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.utils.geometry import find_reciprocal_matches, xy_grid


def depth_normalize(cfg, depth):
    t_min = torch.tensor(cfg.depth_min, device=depth.device)
    t_max = torch.tensor(cfg.depth_max, device=depth.device)

    normalized_depth = (((depth - t_min) / (t_max - t_min)) - 0.5 ) * 2.0

    return normalized_depth

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

    video_path = os.path.join(new_dir, "video.gif")

    # imageio.mimwrite(video_path, vid)
    imageio.mimsave(video_path, vid, 'GIF', fps=1)

    for i, image in enumerate(samples):
        save_image(image, new_dir + f"/frame_{i}.png")


def calculate_psnr(img1, img2):
    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    # Convert images to [0, 255] range if they are in [0, 1] range
    img1 = img1 * 255.0
    img2 = img2 * 255.0

    # Calculate Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)

    if mse == 0:
        return float('inf')  # PSNR is infinite if the images are identical

    # Max pixel value (255 for 8-bit images)
    max_pixel = 255.0

    # Calculate PSNR
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()


def pose_addition(origin_pose, rel_pose):

    o_rot = origin_pose[:3, :3]
    r_rot = rel_pose[:3, :3]

    new_rot = o_rot @ r_rot

    o_loc = origin_pose[:3, -1]
    r_loc = rel_pose[:3, -1]

    new_loc = o_loc + r_loc
    new_mtx = torch.cat((new_rot, new_loc[...,None]),dim=-1)

    fin_mtx = torch.cat((new_mtx, torch.tensor([[0,0,0,1]]).to(o_rot.device)), dim=0)

    return fin_mtx

transform = transforms.ToTensor()


def postprocess_fn(sample):

    # Read the annotation, retrieve views

    instance = sample["__key__"]

    frame_data = sample['annotations'].decode('utf-8').split('\n')[1:-1]
    # frame_keys_list = [f.split(" ")[0] + ".png" for f in frame_data]

    image_list = []
    poses_list = []

    # print(sample.keys())

    for i, single_frame_data in enumerate(frame_data):

        split_data = single_frame_data.split(" ")

        image_key = split_data[0] + ".png"

        try:
            img = transform(sample[image_key])
            image_list.append(img)

            this_data = frame_data[i].split(" ")
            assert this_data[0] == image_key.split(".")[0]

            str_floats = this_data[7:19]
            fl_list = [float(st) for st in str_floats]
            pose = np.array(fl_list).reshape(3,4)
            poses_list.append(pose)
        
        except:
            pass
    
    imgs = torch.stack(image_list)
    poses = np.stack(poses_list)

    output = {"img": imgs, "pose": poses}    
    
    return output


def apply_heatmap(tensor):
    # Ensure tensor is in the right format (1, 1, 512, 512)
    assert tensor.ndim == 4 and tensor.shape[1] == 1, "Tensor must have shape (1, 1, H, W)"
    
    # Remove batch dimension and convert to numpy array
    image_np = tensor[0, 0].cpu().numpy()
    
    # Normalize the tensor to range 0-255 for visualization
    image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply heatmap (COLORMAP_AUTUMN) using OpenCV
    heatmap_np = cv2.applyColorMap(image_np, cv2.COLORMAP_JET)
    
    # Convert back to tensor and add batch dimension
    heatmap_tensor = T.ToTensor()(heatmap_np).unsqueeze(0)
    heatmap_tensor = torch.stack((heatmap_tensor[:,2],heatmap_tensor[:,1],heatmap_tensor[:,0]),dim=1)
    
    return heatmap_tensor

# def compute_depth_variance(depth_map, patch_size=10):
#     """
#     Compute depth variance for each pixel using a local patch.

#     Args:
#         depth_map (torch.Tensor): Depth map tensor of shape (H, W, 1).
#         patch_size (int): Size of the local patch to compute variance (default is 10).
    
#     Returns:
#         torch.Tensor: Depth variance map of shape (H, W, 1).
#     """
#     # Ensure depth_map is of shape (1, 1, H, W) for convolution operations
#     # depth_map = depth_map.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 1, H, W)

#     # Define padding and kernel size
#     pad_size = patch_size // 2
#     kernel_size = patch_size

#     # Pad the depth map
#     depth_map_padded = F.pad(depth_map, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

#     # Create a convolution kernel for mean computation
#     kernel = torch.ones(1, 1, kernel_size, kernel_size, device=depth_map.device) / (kernel_size * kernel_size)

#     # Compute local mean of the depth map
#     local_mean = F.conv2d(depth_map_padded, kernel, stride=1, padding=0)  # Shape: (1, 1, H, W)

#     # Compute local mean of squared depth map
#     depth_squared = depth_map_padded ** 2
#     local_mean_squared = F.conv2d(depth_squared, kernel, stride=1, padding=0)  # Shape: (1, 1, H, W)

#     # Compute variance: E[x^2] - (E[x])^2
#     local_variance = local_mean_squared - local_mean ** 2  # Shape: (1, 1, H, W)

#     # Squeeze dimensions back to (H, W, 1)
#     depth_variance_map = local_variance.squeeze(0).permute(1, 2, 0)  # Shape: (H, W, 1)

#     return depth_variance_map


def compute_depth_variance(depth_map, patch_size=10):
    """
    Compute depth variance for each pixel using a local patch, ignoring zero values.

    Args:
        depth_map (torch.Tensor): Depth map tensor of shape (H, W, 1).
        patch_size (int): Size of the local patch to compute variance (default is 10).
    
    Returns:
        torch.Tensor: Depth variance map of shape (H, W, 1).
    """
    # Ensure depth_map is of shape (1, 1, H, W) for convolution operations
    # depth_map = depth_map.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Define padding and kernel size
    pad_size = patch_size // 2
    kernel_size = patch_size

    # Pad the depth map
    depth_map_padded = F.pad(depth_map, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

    # Create a mask for non-zero pixels
    mask = (depth_map_padded > 0).float()

    # Create a convolution kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=depth_map.device)

    # Compute the sum and count of non-zero pixels in each patch
    sum_depth = F.conv2d(depth_map_padded, kernel, stride=1, padding=0)  # Sum of depth values
    count_nonzero = F.conv2d(mask, kernel, stride=1, padding=0)  # Count of non-zero pixels

    # Avoid division by zero by setting a minimum count of 1
    count_nonzero = torch.clamp(count_nonzero, min=1)

    # Compute local mean of the depth map
    local_mean = sum_depth / count_nonzero

    # Compute sum of squared depth values
    depth_squared = depth_map_padded ** 2
    sum_depth_squared = F.conv2d(depth_squared, kernel, stride=1, padding=0)

    # Compute local mean of squared depth values
    local_mean_squared = sum_depth_squared / count_nonzero

    # Compute variance: E[x^2] - (E[x])^2
    local_variance = local_mean_squared - local_mean ** 2

    # Squeeze dimensions back to (H, W, 1)
    depth_variance_map = local_variance.squeeze(0).permute(1, 2, 0)  # Shape: (H, W, 1)

    return depth_variance_map



def compute_max_within_patch(depth_map, patch_size=10):
    """
    Compute the maximum value for each pixel using a local patch.

    Args:
        depth_map (torch.Tensor): Depth map tensor of shape (H, W, 1).
        patch_size (int): Size of the local patch (default is 10).
    
    Returns:
        torch.Tensor: Max value map of shape (H, W, 1).
    """
    # Ensure depth_map is of shape (1, 1, H, W) for pooling operations
    # depth_map = depth_map.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Apply max pooling
    max_map = -F.max_pool2d(
        -depth_map,
        kernel_size=patch_size,
        stride=1,  # Stride of 1 ensures sliding window over all pixels
        padding=patch_size // 2,  # Padding to handle edges
    )

    # Squeeze dimensions back to (H, W, 1)
    max_map = max_map.squeeze(0).permute(1, 2, 0)  # Shape: (H, W, 1)

    return max_map


def main(cfg):
    
    version = "marigold"
    # version = "naive"
    checkpoint_name = 'marigold_pointmap_warped'
    # checkpoint_name = 'co3d_four_gt_full'
    # checkpoint_name = 'co3d_naive'
    depth_condition = cfg.use_depthmap
    use_mesh = cfg.use_mesh
    use_normal = cfg.use_normal
    use_plucker = cfg.use_plucker
    use_full_gt = False
    use_opt = False
    multitask = cfg.multitask
    
    genwarp_cfg = dict(
        version=version,
        pretrained_model_path='/media/multiview-gen/checkpoints',
        checkpoint_name=checkpoint_name,
        half_precision_weights=False,
        embedder_input_dim=3,
        depth_condition=depth_condition,
        use_mesh=use_mesh,
        use_normal=use_normal,
        use_plucker=use_plucker,
        multitask=multitask,
        ref_expand=cfg.use_ref_expand
    )

    genwarp_nvs = GenWarp(cfg=genwarp_cfg)

    ENDPOINT_URL = 'https://storage.clova.ai'

    os.environ['AWS_ACCESS_KEY_ID'] = "AUIVA2ODFS9S2YDD0A75"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "VDIVVqIC9FCC0GmOQ2nNy3o7NjkWVqC4oTDOz3mM"
    os.environ['S3_ENDPOINT_URL'] = ENDPOINT_URL

    run_setting = "co3d_known"
    # run_setting = "real_known"

    # HERE

    if run_setting == "real_known":
    
        image_dir_0 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec"
        image_dir_1 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/00a7a9c0ea61670d"
        image_dir_2 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a8fda80930b52ae"
        image_dir_3 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a7d39becb0a9771"

        image_dirs = [image_dir_1, image_dir_2, image_dir_3]

        fx = [0, 1, 2]

        frames = []

        for image_dir in image_dirs:

            # Transformation to convert images to tensor and ensure consistent size
            transform = transforms.Compose([
                transforms.Resize((512, 512)),   # Resize to a consistent size (adjust as needed)
                transforms.ToTensor()             # Convert to tensor and normalize to [0, 1]
            ])

            # List to store individual image tensors
            image_tensors = []

            # Read each image in the directory
            for image_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_name)
                if image_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Supported formats
                    image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
                    image_tensor = transform(image)  # Apply transformations
                    image_tensors.append(image_tensor)  # Append to list

            frame_imgs = torch.stack(image_tensors)

            frames.append(frame_imgs)

            transform = transforms.Compose([
                transforms.Resize((512, 512))  # Resize to (512, 512)
            ])

            img_idx = [0, -1]

    elif run_setting == "co3d_known":
    
    
        # image_dir_0 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec"
        image_dir_0 = "/media/data/car/106_12658_23657/images"
        image_dir_1 = "/media/data/car/194_20901_41098/images"
        # image_dir_2 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/co3d_partial/motorcycle/365_39123_75802/images"

        image_dirs = [image_dir_0, image_dir_1]
        fx = [dir.split("/")[-2] for dir in image_dirs]

        # image_dirs = [image_dir_1]

        frames = []

        for image_dir in image_dirs:

            # Transformation to convert images to tensor and ensure consistent size
            transform = transforms.Compose([
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize shortest side to 512
                transforms.CenterCrop(512),  # Center crop to 512x512
                transforms.ToTensor()             # Convert to tensor and normalize to [0, 1]
            ])

            # List to store individual image tensors
            image_tensors = []

            # Read each image in the directory
            for image_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_name)
                if image_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Supported formats
                    image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
                    image_tensor = transform(image)  # Apply transformations
                    image_tensors.append(image_tensor)  # Append to list

            frame_imgs = torch.stack(image_tensors)

            frames.append(frame_imgs)

    # # Run Dust3r
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    # images = load_images(image_files, size=512)

    for x, frame_imgs in enumerate(frames):            
        images = []
        src_images = []
        num_frames = 10
        p = fx[x]

        img_idx = torch.floor(torch.linspace(0, frame_imgs.shape[0]-100, num_frames+2)).int()

        for i, image in enumerate(frame_imgs[img_idx]):
            x = image[None,...]
            img_dict = dict(img = x, true_shape=np.array([[512,512]]), idx=i, instance=str(i))
            images.append(img_dict)
            src_images.append(x)

        # src_images = [transform(to_tensor(Image.open(image_file).convert('RGB'))[None].cuda()) for image_file in image_files]
        # import pdb; pdb.set_trace()

        if not os.path.isfile(f"duster_results/scene_{p}_points.pt"):
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=batch_size)

            # at this stage, you have the raw dust3r predictions
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

            focals = scene.get_focals()
            poses = scene.get_im_poses()
            pts3d = scene.get_pts3d()

            focals = [k.to(device) for k in focals]
            poses = [k.to(device) for k in poses]
            pts3d = [k.to(device) for k in pts3d]

            poses = torch.stack(poses)
            focals = torch.stack(focals)
            points_for_saving = torch.stack(pts3d)
            new_pose = False

            torch.save(points_for_saving,f"duster_results/scene_{p}_points.pt")
            torch.save(poses,f"duster_results/scene_{p}_poses.pt")
            torch.save(focals,f"duster_results/scene_{p}_focals.pt")
        
        else:
            pts3d = torch.load(f"duster_results/scene_{p}_points.pt", map_location="cpu")
            poses = torch.load(f"duster_results/scene_{p}_poses.pt", map_location="cpu")
            focals = torch.load(f"duster_results/scene_{p}_focals.pt", map_location="cpu")
            new_pose=False

        if run_setting == "real_known":
            alphas = poses[1:-1]
        elif run_setting == "co3d_known":
            src_idx = [0, 3, -3, -1]

        t_idx = [i for i in range(len(img_idx)) if i not in src_idx]
        alphas = poses[t_idx]
            
        psnrs = []

        perc = [0.0]

        for masking_percent in perc:

            gt = []
            frames = []
            warped_frames = []
            corresponding_frames = []
            diffmap_1 = []
            diffmap_2 = []
            depth = []
            ult = []

            other_ult = []
            
            pseudo_pts = []
            
            try:
                pseudo_pts_tensor = torch.load(f"{p}_pseudo_pts_tensor.pt")
            except:
                pass

            for idxc, alpha in tqdm(enumerate(alphas)):

                if new_pose == True:
                    eye_pos = (1-alpha) * eye_pos_start + alpha * eye_pos_end
                    lookat = (1-alpha) * lookat_start + alpha * lookat_end

                    src_view_mtx = camera_lookat(
                        torch.tensor([[0., 0., 0.]]).to(device),  # From (0, 0, 0)
                        torch.tensor([[-1., 0., 0.]]).to(device), # Cast rays to -x
                        torch.tensor([[0., 0., 1.]]).to(device)   # z-up
                    ).to(device).float()

                    tar_view_mtx = camera_lookat(
                        eye_pos, # Camera eye position
                        lookat,  # Looking at.
                        torch.tensor([[0., 0., 1.]]).to(device),  # z-up
                    ).to(device).float()

                    rel_view_mtx = (
                        tar_view_mtx.float() @ torch.linalg.inv(src_view_mtx.float().to(device))
                    ).float()

                    target_idx = 1

                    new_target_pose = pose_addition(origin_pose=poses[target_idx], rel_pose=rel_view_mtx[0])
                    
                    src_idx = [k for k in range(len(image_files)) if k is not target_idx]

                else:
                    new_target_pose = alpha
                    target_idx = t_idx[idxc]
                
                # Preparing input information for GenWarp
                
                images = dict(ref=[src_images[k].to(device)[None,...] for k in src_idx], tgt=src_images[target_idx].to(device)[None,...])
                correspondence = dict(ref=torch.cat([pts3d[k][None,...] for k in src_idx])[None,...].to(device), tgt=pts3d[target_idx][None,...].to(device))

                ref_camera=dict(pose=poses[src_idx].to(device)[None,...], 
                                focals=focals[src_idx].to(device)[None,...], 
                                orig_img_size=torch.tensor([512, 512]).to(device))
            
                tgt_camera=dict(pose=new_target_pose[None,...].to(device),
                                focals=focals[target_idx][None,...].to(device),
                                orig_img_size=torch.tensor([512, 512]).to(device))
                
                camera_info = dict(ref=ref_camera, tgt=tgt_camera)
                
                if version != "marigold":
                    mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth = embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, 1, device)
                    tgt_depth_norm, ref_depth = None, None
                elif version == "marigold":
                    tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth = mari_embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, 1, device)
                    
                embedder, out_dim = get_embedder(2)
                dataset = "co3d_duster"
                
                ref_images = images["ref"]
            
                args = dict(
                    src_images=ref_images,
                    correspondence=correspondence,
                    camera_info=camera_info,
                    embedder=embedder,
                    src_idx=src_idx,
                    tgt_idx=target_idx,
                    dataset=dataset,
                    tgt_image=images["tgt"],
                    point_masking=cfg.point_masking,
                    masking_percent=cfg.masking_percent,
                    use_gt_cor=cfg.use_gt_cor,
                    pure_gt=cfg.pure_gt,
                    gt_threshold=cfg.gt_mask_threshold,
                )
                                
                if cfg.use_mesh:
                    args["mesh_pts"] = mesh_pts.to(device)
                    args["mesh_depth"] = mesh_depth.to(device)
                
                if cfg.use_normal:
                    args["mesh_normals"] = mesh_normals.to(device)
                    args["mesh_ref_normals"] = mesh_ref_normals.to(device)

                if cfg.use_depthmap:
                    args["ref_depth"] = norm_depth

                if cfg.use_conf:
                    args["confidence"] = confidence_map

                if cfg.use_plucker:
                    args["plucker"] = plucker

                if cfg.gt_cor_reg:
                    args["gt_cor_regularize"] = True
                    
                pseudo_GT = True
                    
                if pseudo_GT:
                    try:
                        args["pseudo_gt_pts"] = pseudo_pts_tensor[idxc]
                    except:
                        pass

                conditions, render_info = prepare_duster_embedding(**args)
                
                if version == "marigold" or version == "switcher":
                    if cfg.geo_setting == "depth":
                        ref_geo = ref_depth
                        tgt_gt = tgt_depth_norm
                        tgt_geo = mesh_depth
                        
                        warped_depth = mesh_depth.unsqueeze(1).repeat(1,3,1,1)
                        norm_warped_depth = depth_normalize(cfg, warped_depth)
                        
                        clip = True
                        
                        if clip:
                            min_val = -1.0
                            max_val = 1.0
                            norm_warped_depth = torch.clip(norm_warped_depth, min=min_val, max=max_val)
                            
                        tgt_geo = norm_warped_depth
                    
                    elif cfg.geo_setting == "pointmap":
                        ref_pointmaps = correspondence["ref"].reshape(-1,512,512,3).permute(0,3,1,2)
                        tgt_pointmap = correspondence["tgt"].permute(0,3,1,2)
                        tgt_geo = None
                    
                        if cfg.pointmap_norm:
                            minmax_set = True  
                            
                            if minmax_set:
                                ptsmap_min = torch.tensor([-0.7, -0.7, 0.01]).to(device)
                                ptsmap_max = torch.tensor([0.7, 0.3, 1.5]).to(device)
                                
                            else:                      
                                ptsmap_min = ref_pointmaps.permute(0,2,3,1).reshape(-1,3).min(dim=0)[0]
                                ptsmap_max = ref_pointmaps.permute(0,2,3,1).reshape(-1,3).max(dim=0)[0]
                            
                            ref_geo = torch.clip((ref_pointmaps - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                            tgt_gt = torch.clip((tgt_pointmap - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                            
                        if cfg.use_warped_cond:
                            if cfg.use_mesh:
                                tgt_geo = mesh_pts.permute(0,3,1,2)
                            else:
                                print("Warped point cloud not implemented")
                
                elif version == "naive":
                    ref_geo = None
                    tgt_geo = None
                    tgt_gt = None
                
                # save_image(images["tgt"][0],"images_tgt.png")
                
                # save_image(images["ref"][0][0],"images_ref_0.png")
                # save_image(images["ref"][1][0],"images_ref_1.png")
                # save_image(images["ref"][2][0],"images_ref_2.png")
                # save_image(images["ref"][3][0],"images_ref_3.png")
                
                # save_image(renders["warped"],"images_warped.png")
                
                # save_image(tgt_depth_norm * 0.5 + 0.5,"im_tgt_depth.png")
                
                # import pdb; pdb.set_trace()
                
                # normals_rgb = (mesh_normals + 1) / 2
                
                
                if cfg.switcher:
                    # This setting means that image is noised with geo as condition
                    
                    img_task_emb = torch.tensor([0, 1]).float().unsqueeze(0).repeat(1, 1).to(noisy_latents.device)
                    geo_task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(noisy_latents.device)
                    
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)                    
                    noisy_latents = torch.cat([noisy_latents, warped_geo_encoded], dim=1)
                                
                renders = genwarp_nvs(
                    images,
                    conditions,
                    render_info,
                    tgt_gt = tgt_gt,
                    ref_geo = ref_geo,
                    tgt_geo = tgt_geo
                )
                
                # normals_rgb = (mesh_normals + 1) / 2

                # Outputs.
                renders['synthesized']     # Generated image.
                renders['warped']

                whitemask = (renders["warped"][0] != 0).float()

                # diff_2 = torch.sqrt((renders["synthesized"] * whitemask - renders["warped"])**2).sum(dim=1).unsqueeze(1)
                # diff_1 = ((renders["synthesized"] * whitemask - renders["warped"])).sum(dim=1).unsqueeze(1)

                # diff = apply_heatmap(diff_1) * whitemask.to("cpu")
                # diff_2 = apply_heatmap(diff_2) * whitemask.to("cpu")

                # diffmap_1.append(diff[0])
                # diffmap_2.append(diff_2[0])

                diff_gt_ori = torch.sqrt((src_images[target_idx].to(device) - renders["synthesized"].to(device))**2).sum(dim=1).unsqueeze(1)
                diff_gt = apply_heatmap(diff_gt_ori)

                mse = F.mse_loss(src_images[target_idx].to(device), renders["synthesized"].to(device), reduction='mean')
                psnr = 10 * math.log10(1.0 / mse.item())

                psnrs.append(psnr)
                print(psnr)

                # frames.append(renders['synthesized'][0])
                # warped_frames.append(renders['warped'][0])
                gt.append(src_images[target_idx][0])
            
                if cfg.use_depthmap:
                    # import pdb; pdb.set_trace(), dim=-1)
                    z_mask = (renders['tgt_depth'] != 0).float() 
                    depth_img = apply_heatmap(torch.clamp(renders['tgt_depth'] * 3, max=1))
                    depth.append(depth_img[0])
                    # depth_var = apply_heatmap(torch.clamp(depth_var[None,...], max=1))
                    # max_depth = apply_heatmap(torch.clamp(maxpool_depth[None,...] * 3, max=1))
                    
                if cfg.use_mesh:
                    depth_img = apply_heatmap(torch.clamp(mesh_depth[None,...], max=1))
                    mesh_normals = mesh_normals.permute(0,3,1,2).to(device) * 0.5 + 0.5

                space = torch.ones(3,512,80).to(device)

                # import pdb; pdb.set_trace()

                if not cfg.use_mesh:
                    if not depth_condition:
                        # import pdb; pdb.set_trace()
                        stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1).detach()
                    else:
                        stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, depth_img[0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1).detach()
                        # stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['correspondence'][0].to(device), space, depth_img[0].to(device), space, depth_var[0].to(device), space, max_depth[0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1)
                else:
                    # import pdb;
                    # stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, mesh_normal_mask[0].permute(2,0,1).repeat(3,1,1).to(device), space, mesh_normals[0], space, z_mask[0] * depth_img[0].to(device), space, renders['synthesized'][0].to(device)), dim=-1).detach()
                    stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, mesh_normal_mask[0].permute(2,0,1).repeat(3,1,1).to(device), space, mesh_normals[0], space, z_mask[0] * depth_img[0].to(device), space, renders['synthesized'][0].to(device), space, (tgt_gt[0] * 0.5 + 0.5).to(device)), dim=-1).detach()
                    
                if version == "marigold":
                    # import pdb; pdb.set_trace()
                    pseudo_pts.append((ptsmap_max[None,None,None,...] - ptsmap_min[None,None,None,...]) * renders["synthesized"].permute(0,2,3,1) + ptsmap_min[None,None,None,...])

                ult.append(stack_img)

                # if renders["other"] is not None:
                #     other_diff = torch.sqrt((renders["other"].to(device) - renders["correspondence"].to(device))**2).sum(dim=1).unsqueeze(1)
                #     other_diff = apply_heatmap(other_diff)
                #     oth_stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device), space, renders["correspondence"][0].to(device), space, renders["other"][0].to(device), space, other_diff[0].to(device)), dim=-1)

                #     other_ult.append(oth_stack_img)


                # import pdb; pdb.set_trace()
                # corresponding_frames.append(renders["correspondence"][0])

            now = strftime("%m_%d_%H_%M_%S", gmtime())
            exp_name = checkpoint_name
            image_dir = os.path.join("outputs", f"{now}/{exp_name}_reference_frames.png")

            # import pdb; pdb.set_trace()
            # make_video(frames, now, folder_name="syn_mask_yes_norm")
            # make_video(gt, now, folder_name="gt_frames")
            # make_video(warped_frames, now, folder_name="warp")
            # make_video(corresponding_frames, now, folder_name="corres")
            make_video(ult, now, folder_name="everything")

            # if len(other_ult) >= 1:
            #     make_video(other_ult, now, folder_name="multi_results")

            # # make_video(diffmap_1, now, folder_name="diff_1")
            # # make_video(diffmap_2, now, folder_name=f"diff_2_{masking_percent}")
            # import pdb; pdb.set_trace()
            save_image(torch.cat(images["ref"]).squeeze(), image_dir)
            # print(psnrs)    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="././train_configs/marigold_pointmap_train_co3d.yaml")
    # parser.add_argument("--config", type=str, default="././train_configs/train_co3d_gt.yaml")
    args = parser.parse_args()
    
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    
    main(config)
