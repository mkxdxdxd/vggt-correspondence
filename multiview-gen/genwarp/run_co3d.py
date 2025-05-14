import os
import sys
import glob
import torch
import numpy as np
import imageio
import random
import json

from PIL import Image
from time import gmtime, strftime
from tqdm import tqdm

from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from genwarp import GenWarp
from genwarp.ops import camera_lookat, get_projection_matrix

from genwarp.utils.projector import ndc_rasterizer

from co3d_dataset import Co3DDataset

import torchvision.transforms as transforms

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import open3d as o3d
from typing import cast

# from pointcloud_loader import get_rgbd_point_cloud

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.utils.geometry import find_reciprocal_matches, xy_grid


HOLDOUT_CATEGORIES = set([
    'carrot'
])

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


def get_rays(H, W, focals, c2w):
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

    # import pdb; pdb.set_trace()

    # short_idx = torch.min(torch.tensor([H,W]))
    # factor = 512 * max(H,W)/min(H,W)

    ray_len = 512
    short_side = torch.min(H,W)

    ray_W = ray_len * W / short_side
    ray_H = ray_len * H / short_side

    margin_W = ray_W / 2 - ray_len / 2
    margin_H = ray_H / 2 - ray_len / 2

    i, j = torch.meshgrid(torch.arange(ray_len, dtype=torch.float32), torch.arange(ray_len, dtype=torch.float32), indexing='xy')

    # Compute directions (normalized by focal length)
    dirs_stack = []
    for focal in focals:
        dirs = torch.stack([(i - ray_W * 0.5 + margin_W) / focal, -(j - ray_H * 0.5 + margin_H) / focal, -torch.ones_like(i)], dim=-1)  # Shape (H, W, 3)
        dirs_stack.append(dirs)

    dirs = torch.stack(dirs_stack)

    # import pdb; pdb.set_trace()

    # Apply camera-to-world rotation matrix to directions
    rays_d = torch.sum(dirs[..., None, :] * c2w[:, None, None, :3, :3], dim=-1)  # Shape (H, W, 3)

    # Broadcast ray origins to match shape (H, W, 3)
    rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)  # Shape (H, W, 3)

    return rays_o, rays_d



def pose_to_ray(camera_info, depth, img_size, batch_size=1, num_viewpoints=1):

    R = camera_info["rot"]
    t = camera_info["trans"]
    focals = camera_info["intrinsic"][:,0,0]

    cam_center = -torch.matmul(R, t)  # (N, 3, 1)
    rot = R[None,...] * torch.tensor([-1,1,-1])[None, None, None, :]

    # import pdb; pdb.set_trace()

    camera_poses = torch.cat((rot, cam_center[None,...]),dim=-1) 
    homogeneous = torch.tensor([0,0,0,1]).to(R.device).expand(batch_size, num_viewpoints, 1, 4)

    c2w = torch.cat((camera_poses, homogeneous), dim=-2)[0]
    rays_o, rays_d = get_rays(img_size[0], img_size[1], focals, c2w)

    depths = depth.permute(0,2,3,1)

    pts_loc = depths * rays_d + rays_o
    unprojected_points = pts_loc[:3].reshape(-1,3)

    return unprojected_points


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


if __name__ == "__main__":

    genwarp_cfg = dict(
        pretrained_model_path='./checkpoints',
        checkpoint_name='multi1',
        half_precision_weights=False
    )

    genwarp_nvs = GenWarp(cfg=genwarp_cfg)

    co3d_path = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/co3d_partial"
    category = "car"

    folders = glob.glob(os.path.join(co3d_path, category, "*/"))
    folders = [os.path.basename(os.path.normpath(folder)) for folder in folders]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=512),
        transforms.CenterCrop(size=(512,512))  # Example cropping to 224x224
    ])

    # Create the dataset with multiple random samples per call (e.g., 3 random samples)
    num_viewpoints = 4
    num_ref_viewpoints = 3
    batch_size = 1

    dataset = Co3DDataset(co3d_path=co3d_path, num_random_samples=num_viewpoints, transform=transform)

    # Create the DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")

        img_size = batch["orig_img_size"][0,0]
        focals = batch["K"][0,:,0,0]

        target_idx = 3
        src_idx = [k for k in range(num_viewpoints) if k is not target_idx]

        images = dict(ref=[batch["image"][0,k][None,...] for k in src_idx], tgt=batch["image"][0,target_idx][None,...])

        ref_camera = dict(rot=batch["R"][0,src_idx],
                          trans=batch["T"][0,src_idx] ,
                          intrinsic=batch["K"][0,src_idx],
                          focals=batch["focal_length"][0,src_idx],
                          orig_img_size=batch["orig_img_size"][0,src_idx])       

        target_idx = src_idx + [target_idx] 

        tgt_camera = dict(rot=batch["R"][0,target_idx],
                          trans=batch["T"][0,target_idx] ,
                          intrinsic=batch["K"][0,target_idx],
                          focals=batch["focal_length"][0,target_idx],
                          orig_img_size=batch["orig_img_size"][0,target_idx])
        
        camera_info = dict(ref=ref_camera, tgt=tgt_camera)

        mask_depth = False
        depth =  batch["depth"][0,src_idx]

        if mask_depth:
            depth = batch["mask"][0,src_idx] * depth

        points = pose_to_ray(ref_camera, depth, img_size, num_viewpoints=num_ref_viewpoints)
        correspondence = dict(ref=points, tgt=None)

        renders = genwarp_nvs(
            images = images,
            correspondence = correspondence,
            camera_info = camera_info,
            ndc=True
        )


    # # Run Dust3r
    # device = 'cuda'
    # batch_size = 1
    # schedule = 'cosine'
    # lr = 0.01
    # niter = 300

    # transform = transforms.Compose([
    #     transforms.Resize((512, 512))  # Resize to (512, 512)
    # ])

    # model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # # you can put the path to a local checkpoint in model_name if needed
    # model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # # load_images can take a list of images or a directory
    # images = load_images(image_files, size=512)

    # src_images = [transform(to_tensor(Image.open(image_file).convert('RGB'))[None].cuda()) for image_file in image_files]

    # for i, img in enumerate(images):
    #     img["img"] = transform(img['img'])
    #     img["true_shape"] = np.array([[512,512]])

    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    # output = inference(pairs, model, device, batch_size=batch_size)

    # # at this stage, you have the raw dust3r predictions
    # view1, pred1 = output['view1'], output['pred1']
    # view2, pred2 = output['view2'], output['pred2']

    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # focals = scene.get_focals()
    # poses = scene.get_im_poses()
    # pts3d = scene.get_pts3d()

    # focals = [k.to(device) for k in focals]
    # poses = [k.to(device) for k in poses]
    # pts3d = [k.to(device) for k in pts3d]

    # For camera transition video

    # eye_pos_start = torch.tensor([[0., 0., 0.]]).to(device)
    # lookat_start = torch.tensor([[-1., 0., 0.]]).to(device)

    # eye_pos_end = torch.tensor([[0.02, -0.02, 0.02]]).to(device)
    # lookat_end = torch.tensor([[-0.7, -0.2, 0.]]).to(device)

    # num_frames = 20

    # alphas = torch.linspace(0, 1, steps=num_frames).to(device)

    # frames = []
    # warped_frames = []

    # for alpha in tqdm(alphas):

        # eye_pos = (1-alpha) * eye_pos_start + alpha * eye_pos_end
        # lookat = (1-alpha) * lookat_start + alpha * lookat_end

        # src_view_mtx = camera_lookat(
        #     torch.tensor([[0., 0., 0.]]).to(device),  # From (0, 0, 0)
        #     torch.tensor([[-1., 0., 0.]]).to(device), # Cast rays to -x
        #     torch.tensor([[0., 0., 1.]]).to(device)   # z-up
        # ).to(device).float()

        # tar_view_mtx = camera_lookat(
        #     eye_pos, # Camera eye position
        #     lookat,  # Looking at.
        #     torch.tensor([[0., 0., 1.]]).to(device),  # z-up
        # ).to(device).float()

        # rel_view_mtx = (
        #     tar_view_mtx.float() @ torch.linalg.inv(src_view_mtx.float().to(device))
        # ).float()

        # target_idx = 1

        # new_target_pose = pose_addition(origin_pose=poses[target_idx], rel_pose=rel_view_mtx[0])
        
        # src_idx = [k for k in range(len(image_files)) if k is not target_idx]



    # Outputs.
    renders['synthesized']     # Generated image.
    renders['warped']

    frames.append(renders['synthesized'][0])
    warped_frames.append(renders['warped'][0])

    now = strftime("%m_%d_%H_%M_%S", gmtime())
    make_video(frames, now, folder_name="syn_mask_yes_norm")
    make_video(warped_frames, now, folder_name="warp")






