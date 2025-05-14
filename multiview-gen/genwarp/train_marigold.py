import argparse
import logging
import math
import os
import os.path as osp
from os.path import join
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import wandb
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
# import open3d as o3d
import multiprocessing as mp

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
import random
import webdataset as wds
import time
from PIL import Image
# from genwarp import GenWarp


from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from genwarp.models.resnet import InflatedConv3d, InflatedGroupNorm
from genwarp.models.mutual_self_attention import ReferenceAttentionControl
from genwarp.models.pose_guider import PoseGuider
from genwarp.models.unet_2d_condition import UNet2DConditionModel
from genwarp.models.unet_3d import UNet3DConditionModel
from genwarp.models.hook import UNetCrossAttentionHooker, XformersCrossAttentionHooker
from genwarp.pipelines.pipeline_nvs import NVSPipeline
from training_utils import delete_additional_ckpt, import_filename, seed_everything, load_model
from einops import rearrange

from training_utils import forward_warper, camera_controller, plucker_embedding, get_embedder, get_coords
from torchvision.transforms.functional import to_pil_image

from lora_diffusion import inject_trainable_lora, extract_lora_ups_down

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

# from genwarp.utils.projector import ndc_rasterizer

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
    compute_plucker_embed,
    postprocess_co3d,
    postprocess_realestate,
    PointmapNormalizer,
    UncertaintyLoss
)

from train_model import prepare_duster_embedding

from data.co3d_dataset import Co3DDataset
from realestate_dataset import RealEstateDataset

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from functools import partial

# from torch.nn.parallel import DistributedDataParallel as DDP

# real10k
from dataset.preprocessed_re10k import PreprocessedRe10k
# megascenes
from dataset.preprocessed_megascenes import PreprocessedMegaScenes

from transformers import CLIPImageProcessor
from training_utils import pred_depth_inference

from dataset.scannet import Scannetdataset

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# from OpenGL import osmesa


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        ref_coord_embeds,
        tgt_coord_embed,
        uncond_fwd: bool = False,
        correspondence = None,
        weight_dtype = None,
        gt_target_coord_embed = None,
        task_emb = None
    ):

        ref_cond_latents = []
        num_viewpoints = len(ref_coord_embeds)

        for ref_embed in ref_coord_embeds:
            ref_cond_tensor = ref_embed.to(device="cuda").to(weight_dtype).unsqueeze(2)
            ref_cond = self.pose_guider(ref_cond_tensor)

            ref_cond_latents.append(ref_cond[:,:,0,...])
        
        # if depth_cond:
        tgt_cond_tensor = tgt_coord_embed.to(device="cuda").unsqueeze(2)
        tgt_cond_latent = self.pose_guider(tgt_cond_tensor)
        tgt_cond_latent = tgt_cond_latent[:,:,0,...]

        if gt_target_coord_embed != None:
            gt_tgt_cond_tensor = gt_target_coord_embed.to(device="cuda").unsqueeze(2)
            gt_tgt_cond_latent = self.pose_guider(gt_tgt_cond_tensor)
            gt_tgt_cond_latent = gt_tgt_cond_latent[:,:,0,...]

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for i, ref_latent in enumerate(ref_image_latents):
                self.reference_unet(
                    ref_latent,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds[i],
                    pose_cond_fea=ref_cond_latents[i],
                    return_dict=False,
                    reference_idx=i,
                )

            self.reference_control_reader.update(self.reference_control_writer, correspondence=correspondence)

        if gt_target_coord_embed != None:
            with torch.no_grad():
                _ = self.denoising_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=clip_image_embeds[0],
                    pose_cond_fea=gt_tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
                ).sample
            
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds[0],
            pose_cond_fea=tgt_cond_latent.unsqueeze(2), #TODO : temporarily can be disabled for debugging
            class_labels=task_emb
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    
    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def log_validation(
    net,
    accelerator,
    ref_reader,
    ref_writer,
    feature_fusion_type='attention_full_sharing',
    multitask = False,
    depth_condition=False,
    use_mesh=False,
    use_normal=False,
    use_plucker=False
):
    logger.info("Running validation... ")

    is_warped_feat_injection = feature_fusion_type == 'warped_feature'
    ori_net = accelerator.unwrap_model(net)

    genwarp_cfg = dict(
        pretrained_model_path='./checkpoints',
        half_precision_weights=False,
        embedder_input_dim=3,
        depth_condition=depth_condition,
        use_mesh=use_mesh,
        use_normal=use_normal,
        use_plucker=use_plucker,
        training_val=True,
        multitask=multitask
    )

    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider

    # input_conv = InflatedConv3d(
    #     4, 320, kernel_size=3, padding=(1, 1)
    # ) 
    # output_conv = InflatedConv3d(
    #         320, 4, kernel_size=3, padding=(1, 1)
    # )

    # input_conv.weight = torch.nn.Parameter(denoising_unet.conv_in.weight[:,:4])
    # output_conv.weight = torch.nn.Parameter(denoising_unet.conv_out.weight[:4])

    # input_conv.bias = denoising_unet.conv_in.bias
    # output_conv.bias = torch.nn.Parameter(denoising_unet.conv_out.bias[:4])

    # denoising_unet.conv_in = input_conv
    # denoising_unet.conv_out = output_conv

    genwarp_nvs = GenWarp(cfg=genwarp_cfg, reference_unet=reference_unet,
                          denoising_unet=denoising_unet, pose_guider=pose_guider,
                          ref_reader=ref_reader, ref_writer=ref_writer)

    generator = torch.manual_seed(42)

    # image_dir_0 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/co3d_partial/car/106_12658_23657/images"
    image_dir_1 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/co3d_partial/car/194_20901_41098/images"
    image_dir_2 = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/co3d_partial/motorcycle/365_39123_75802/images"

    image_dirs = [image_dir_1, image_dir_2]
    ps = [1, 2]

    frames = []
    psnr_list = []

    for image_dir in image_dirs:

        # Transformation to convert images to tensor and ensure consistent size
        transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.Resize((512, 512)),   # Resize to a consistent size (adjust as needed)
            transforms.ToTensor()             # Convert to tensor and normalize to [0, 1]
        ])

        # List to store individual image tensors
        image_tensors = []
        target_imgs = []

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

        img_tiles = []

        for x, frame_imgs in enumerate(frames):            
            images = []
            src_images = []
            num_frames = 10
            p = ps[x]

            img_idx = torch.floor(torch.linspace(0, frame_imgs.shape[0]-1, num_frames+2)).int()

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

            src_idx = [0, 3, 6, -1]

            t_idx = [i for i in range(len(img_idx)) if i not in src_idx]
            alphas = poses[t_idx]
                
            frames = []

            gt_imgs = []
            syn_imgs = []

            for idxc, alpha in tqdm(enumerate(alphas)):

                new_target_pose = alpha
                target_idx = t_idx[idxc]
                
                # Preparing input information for GenWarp
                images = dict(ref=[src_images[k].to(device) for k in src_idx], tgt=src_images[target_idx].to(device))
                correspondence = dict(ref=torch.cat([pts3d[k][None,...].to(device) for k in src_idx]), tgt=pts3d[target_idx])

                ref_camera=dict(pose=poses[src_idx].to(device), 
                                focals=focals[src_idx].to(device), 
                                orig_img_size=torch.tensor([512, 512]).to(device))
            
                tgt_camera=dict(pose=new_target_pose[None,...].to(device),
                                focals=focals[target_idx][None,...].to(device),
                                orig_img_size=torch.tensor([512, 512]).to(device))
                
                camera_info = dict(ref=ref_camera, tgt=tgt_camera)

                downsample = False
                downsample_by = 0
            
                if use_mesh:
                    if downsample:
                        downsample_by = downsample_by
                        start = downsample_by // 2
                        interval = downsample_by
                        points = correspondence["ref"][:,start::interval,start::interval,:].permute(0,3,1,2)
                        rgb = torch.ones_like(points)
                        side_length = 512 // downsample_by

                    else:
                        points = correspondence["ref"].permute(0,3,1,2)
                        rgb = torch.ones_like(points)
                        side_length = 512

                    pts_list = points
                    color_list = rgb
                    orig_length = torch.tensor(512).to(device)

                    mesh_pts = []
                    mesh_normals = []
                    mesh_depth = []
                    mesh_ref_normal_list = []

                    extrins = tgt_camera["pose"][0].detach().cpu().numpy()
                    focal_length = tgt_camera["focals"][0]

                    vert = []
                    fc = []
                    col = []

                    vert_stack = 0
                    not_original_mesh = True

                    # if original_mesh:

                    for k, (pts, color) in enumerate(zip(pts_list, color_list)):

                        vertices, faces, colors = features_to_world_space_mesh(
                            world_space_points=pts.detach(),
                            colors=color.detach(),
                            edge_threshold=0.48,
                            H = side_length
                        )

                        vert.append(vertices)
                        fc.append(faces + vert_stack)
                        col.append(colors)

                        vert_num = vertices.shape[1]
                        vert_stack += vert_num

                    vertices = torch.cat(vert, dim=-1)
                    faces = torch.cat(fc, dim=-1)
                    colors = torch.cat(col, dim=-1)
                    inv_extrins = np.linalg.inv(extrins)

                    mesh, o3d_device = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)
                    rendered_depth, normals = mesh_rendering(mesh, focal_length, inv_extrins, o3d_device)
                    
                    rays_o, rays_d = get_rays(orig_length, orig_length, focal_length, torch.tensor(extrins).to(device), 1, device)
                    mask = (rendered_depth != 0)  

                    proj_pts = mask[...,None].to(device) * (rays_o[0,0] + rendered_depth[...,None].to(device) * rays_d[0,0])

                    mesh_pts.append(proj_pts)
                    mesh_depth.append(rendered_depth)
                    mesh_normals.append(normals)

                    if use_normal:
                        per_batch_ref_normals = []

                        for ref_extrins in ref_camera["pose"]:
                            ref_extrins = ref_extrins.detach().cpu().numpy()
                            ref_pose = np.linalg.inv(ref_extrins)
                            ref_depths, ref_normals = mesh_rendering(mesh, focal_length, ref_pose, o3d_device)
                            per_batch_ref_normals.append(ref_normals.permute(2,0,1))
                        
                        mesh_ref_normals = torch.stack(per_batch_ref_normals)
                        mesh_ref_normal_list.append(mesh_ref_normals)

                    mesh_pts = torch.stack(mesh_pts)[None,...]
                    mesh_depth = torch.stack(mesh_depth)[None,...]
                    mesh_normals = torch.stack(mesh_normals)[None,...]

                    if use_normal:
                        mesh_ref_normals = torch.stack(mesh_ref_normal_list)

                if use_plucker:
                    orig_length = torch.tensor(512).to(device)
                    rays_o, rays_d = get_rays(orig_length, orig_length, ref_camera["focals"], ref_camera['pose'], batch_size, device)
                    ref_plucker_embed = compute_plucker_embed(rays_o, rays_d, batch_size, 512, 512, ref_camera["pose"])

                    rays_o, rays_d = get_rays(orig_length, orig_length, tgt_camera["focals"].unsqueeze(1), tgt_camera['pose'].unsqueeze(1), batch_size, device)
                    tgt_plucker_embed = compute_plucker_embed(rays_o, rays_d, 1, 512, 512, tgt_camera["pose"].unsqueeze(1))

                    plucker = {
                        "ref": ref_plucker_embed.to(device),
                        "tgt": tgt_plucker_embed.to(device)
                    }                    

                if depth_condition:
                    origins = ref_camera['pose'][:,:3,-1]
                    ref_depth = torch.linalg.norm(correspondence["ref"] - origins[...,None,None,:],axis=-1)[...,None]

                if use_mesh:
                    correspondence["ref"] = points.permute(0,2,3,1)

                args = dict(images = images,
                    correspondence = correspondence,
                    camera_info = camera_info,
                    )
            
                if depth_condition:
                    args["ref_depth"] = ref_depth 
                    args["depth_condition"] = depth_condition

                if use_normal:
                    args["mesh_normals"] = mesh_normals.to(device)
                    args["mesh_ref_normals"] = mesh_ref_normals.to(device)

                if use_mesh:
                    args["mesh_pts"] = mesh_pts.to(device)
                    args["mesh_depth"] = mesh_depth.to(device)

                if use_plucker:
                    args["plucker"] = plucker

                renders = genwarp_nvs(
                    **args
                )

                gen_img = renders['synthesized'] 

                max_pixel_value = 1.0
                mse = F.mse_loss(src_images[target_idx].to(device), gen_img, reduction='mean')
                psnr = 10 * torch.log10(max_pixel_value**2 / mse)

                psnr_list.append(psnr)

                gt_imgs.append(src_images[target_idx].to(device))
                syn_imgs.append(gen_img)

            images = torch.cat((torch.cat(gt_imgs), torch.cat(syn_imgs)), dim=-2)
        
        img_tiles.append(images)
    
    psnr_mean = torch.mean(torch.tensor(psnr_list))

    return psnr_mean, img_tiles


def depth_normalize(cfg, depth):
    t_min = torch.tensor(cfg.depth_min, device=depth.device)
    t_max = torch.tensor(cfg.depth_max, device=depth.device)

    normalized_depth = (((depth - t_min) / (t_max - t_min)) - 0.5 ) * 2.0

    return normalized_depth


def encode_depth(depth, vae, weight_dtype):
    # Depth: (B, H, W, 1)

    normalized_depth = depth_normalize(depth)
    stacked_depth = normalized_depth.repeat(1,1,1,3).permute(0, 3, 1, 2)
    
    latent_depth = vae.encode(stacked_depth.to(weight_dtype)).latent_dist.sample()

    return latent_depth

    
@torch.no_grad()
def mari_embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, batch_size, device, current_dataset = None):
    
    mesh_pts = None
    mesh_depth = None
    mesh_normals = None
    mesh_ref_normals = None
    mesh_normal_mask = None
    norm_depth = None
    confidence_map = None
    plucker = None 
    tgt_depth = None
    
    origins = ref_camera['pose'][:,:,:3,-1]
    tgt_origins = tgt_camera['pose'][:,:3,-1]
    
    dist = torch.linalg.norm(correspondence["ref"] - origins[...,None,None,:],axis=-1)[...,None]
    norm_depth = dist
    
    tgt_dist = torch.linalg.norm(correspondence["tgt"] - tgt_origins[...,None,None,:],axis=-1)[...,None]
    tgt_depth = tgt_dist
                                
    ref_depth = depth_normalize(cfg, norm_depth)
    tgt_depth_norm = depth_normalize(cfg, tgt_depth)
    
    clip = True
                                
    if clip:
        min_val = -1.0
        max_val = 1.0
        ref_depth = torch.clip(ref_depth, min=min_val, max=max_val).reshape(-1,512,512,1).permute(0,3,1,2).repeat(1,3,1,1)
        tgt_depth_norm = torch.clip(tgt_depth_norm, min=min_val, max=max_val).permute(0,3,1,2).repeat(1,3,1,1)
    
    # import pdb; pdb.set_trace()

    # if cfg.use_mesh:
    if cfg.downsample:
        downsample_by = cfg.downsample_by
        start = downsample_by // 2
        interval = downsample_by
        points = correspondence['ref'][:,:,start::interval,start::interval,:].permute(0,1,4,2,3).float()            
        images_ref = torch.cat(images["ref"], dim=1)
        rgb = images_ref[:,:,:,start::interval,start::interval]
        side_length = 512 // downsample_by

    # else:
    #     points = batch['points'].reshape(batch_size, -1, 3).permute(0,2,1).float()
    #     rgb = batch['image'].permute(0,1,3,4,2).reshape(batch_size, -1, 3).permute(0,2,1)
    #     side_length = 512 // downsample_by

    batch_pts = points
    batch_colors = rgb
    orig_length = torch.tensor(512).to(device)

    mesh_pts = []
    mesh_normals = []
    mesh_depth = []
    mesh_ref_normal_list = []
    mesh_normal_mask = []

    for i, (pts_list, color_list) in enumerate(zip(batch_pts, batch_colors)):
        extrins = tgt_camera["pose"][i].detach().cpu().numpy()
        focal_length = tgt_camera["focals"][i]

        vert = []
        fc = []
        col = []

        vert_stack = 0

        for k, (pts, color) in enumerate(zip(pts_list, color_list)):
            # import pdb; pdb.set_trace()

            vertices, faces, colors = features_to_world_space_mesh(
                world_space_points=pts.detach(),
                colors=color.detach(),
                edge_threshold=0.48,
                H = side_length
            )

            vert.append(vertices)
            fc.append(faces + vert_stack)
            col.append(colors)

            vert_num = vertices.shape[1]
            vert_stack += vert_num

        vertices = torch.cat(vert, dim=-1)
        faces = torch.cat(fc, dim=-1)
        colors = torch.cat(col, dim=-1)

        mesh, o3d_device = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)

        inv_extrins = np.linalg.inv(extrins)
        rendered_depth, normals = mesh_rendering(mesh, focal_length, inv_extrins, o3d_device)
        rays_o, rays_d = get_rays(orig_length, orig_length, focal_length.to(device), torch.tensor(extrins).to(device), 1, device)
        mask = (rendered_depth != 0)

        proj_pts = mask[...,None].to(device) * (rays_o[0,0] + rendered_depth[...,None].to(device) * rays_d[0,0])

        if cfg.use_normal_mask and current_dataset != "realestate":
            center_dir = rays_d[0, 0, 256, 256][None,None,...] 

            normed_center_dir = -center_dir / torch.norm(center_dir, dim=-1, keepdim=True) 
            normed_normals = normals.to(device) / torch.norm(normals, dim=-1, keepdim=True).to(device)

            dot_product = torch.clamp(torch.sum(normed_center_dir * normed_normals, dim=-1, keepdim=True), -1.0, 1.0) 
            angle_difference = torch.acos(dot_product)

            angle_mask = angle_difference > (torch.pi * 1/2)
            mesh_normal_mask.append(angle_mask)

        mesh_pts.append(proj_pts)
        mesh_depth.append(rendered_depth)
        mesh_normals.append(normals)

        if cfg.use_normal:
            per_batch_ref_normals = []

            for ref_extrins, ref_focal in zip(ref_camera["pose"][i], ref_camera["focals"][i]):
                ref_extrins = ref_extrins.detach().cpu().numpy()
                ref_pose = np.linalg.inv(ref_extrins)
                ref_depths, ref_normals = mesh_rendering(mesh, ref_focal, ref_pose, o3d_device)
                per_batch_ref_normals.append(ref_normals.permute(2,0,1))
            
            mesh_ref_normals = torch.stack(per_batch_ref_normals)
            mesh_ref_normal_list.append(mesh_ref_normals)
        
    mesh_pts = torch.stack(mesh_pts)
    mesh_depth = torch.stack(mesh_depth).to(device)
    mesh_normals = torch.stack(mesh_normals)

    if cfg.use_normal:
        mesh_ref_normals = torch.stack(mesh_ref_normal_list)

    if cfg.use_normal_mask and current_dataset != "realestate":
        # save_image(torch.cat((mesh_normals.permute(0,3,1,2).to(device), mesh_normal_mask[0][None,...].permute(0,3,1,2).repeat(1,3,1,1))), "new.png")

        mesh_normal_mask = (1 - torch.stack(mesh_normal_mask).float()).to(device)

        mesh_pts = mesh_pts * mesh_normal_mask
        mesh_depth = mesh_depth * mesh_normal_mask[...,0]
        mesh_normals = mesh_normals.to(device) * mesh_normal_mask
            
    return tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth


def mesh_get_depth(pts, color, extrins, focal_length, side_length, device):

    vertices, faces, colors = features_to_world_space_mesh(
        world_space_points=pts.detach(),
        colors=color.detach(),
        edge_threshold=0.48,
        H = side_length
    )

    mesh, o3d_device = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)

    inv_extrins = np.linalg.inv(extrins)

    depth, normals = mesh_rendering(mesh, focal_length, inv_extrins, o3d_device)

    return depth, normals



def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_params)
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    if cfg.debugging:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
            mixed_precision=cfg.solver.mixed_precision,
            kwargs_handlers=[kwargs],
        )    

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
            mixed_precision=cfg.solver.mixed_precision,
            log_with="wandb",
            # project_dir="./mlruns",
            kwargs_handlers=[kwargs],
            # dispatch_batches = False,
        )

    # print(accelerator.use_distributed)
    # print(accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    from datetime import datetime
    now = datetime.now()
    formatted_now = now.strftime("%y%m%d_%H%M%S")
    
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}_{formatted_now}"
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    cfg.save_dir = save_dir
    
    # cfg save to yaml
    with open(f"{save_dir}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    
    # val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    # Option whether to start training from scratch or not
    train_from_scratch = cfg.train_from_scratch

    cond_channels = 16
    if cfg.use_depthmap:
        cond_channels += 1
    if cfg.use_conf:
        cond_channels += 1
    if cfg.use_normal:
        cond_channels += 3
    if cfg.use_plucker:
        cond_channels += 6

    # Pose guider.
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320,
        conditioning_channels=cond_channels,
    ).to(device="cuda", dtype=weight_dtype)
        
    if not train_from_scratch:
        # Reference Unet.
        reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                join(cfg.model_path, 'config.json')
        )).to(device="cuda", dtype=weight_dtype)

        reference_unet.load_state_dict(torch.load(
            join(cfg.model_path, 'reference_unet.pth'),
            map_location= 'cpu'),
        )

        # Denoising Unet.
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            join(cfg.model_path, 'denoising_config.json'),
            join(cfg.model_path, 'denoising_unet.pth')
        ).to(device="cuda", dtype=weight_dtype)

        # if cfg.multitask and not cfg.switcher:
        if denoising_unet.conv_in.weight.shape[1] == 8:
            if cfg.use_warped_cond:
                expand_weights = denoising_unet.conv_in.weight[:,4:] 
                inflated_weights = torch.cat((denoising_unet.conv_in.weight, expand_weights), dim=1) * (2/3)
                
                input_conv = InflatedConv3d(
                    12, 320, kernel_size=3, padding=(1, 1)
                ) 
                
                input_conv.weight = torch.nn.Parameter(inflated_weights)
                input_conv.bias = denoising_unet.conv_in.bias 
                
                denoising_unet.conv_in = input_conv

        if reference_unet.conv_in.weight.shape[1] == 8:
            if cfg.use_ref_expand:
                ref_expand_weights = reference_unet.conv_in.weight[:,4:] 
                inflated_ref_in_weight = torch.cat([reference_unet.conv_in.weight, ref_expand_weights], dim=1) * (2/3)

                input_conv = nn.Conv2d(
                    12, 320, kernel_size=3, padding=(1, 1)
                ) 

                input_conv.weight = torch.nn.Parameter(inflated_ref_in_weight)
                input_conv.bias = reference_unet.conv_in.bias 

                reference_unet.conv_in = input_conv

        try:
            pose_guider.load_state_dict(torch.load(
                join(cfg.model_path, 'pose_guider.pth'),
                map_location='cpu'),
            )
            
        except:
            pass

    else:
        reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                join(cfg.model_path_scratch, 'config.json')
        )).to(device="cuda", dtype=weight_dtype)

        ref_param_names = [ name for name,_ in reference_unet.named_parameters()]
        loading_params = {key : param for key, param in torch.load(join(cfg.model_path_scratch, 'diffusion_pytorch_model.bin'),map_location= 'cpu').items()
                        if key in ref_param_names}
        reference_unet.load_state_dict(loading_params)
                                
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            join(cfg.model_path_scratch, 'config.json'),
            join(cfg.model_path_scratch, 'diffusion_pytorch_model.bin')
        ).to(device="cuda", dtype=weight_dtype)
        
        if cfg.use_warped_cond:
            expand_weights = denoising_unet.conv_in.weight[:,4:] 
            inflated_weights = torch.cat((denoising_unet.conv_in.weight, expand_weights), dim=1) * (2/3)
            
            input_conv = InflatedConv3d(
                12, 320, kernel_size=3, padding=(1, 1)
            ) 
            
            input_conv.weight = torch.nn.Parameter(inflated_weights)
            input_conv.bias = denoising_unet.conv_in.bias 
            
            denoising_unet.conv_in = input_conv
            
        if cfg.use_ref_expand:
            ref_expand_weights = reference_unet.conv_in.weight[:,4:] 
            inflated_ref_in_weight = torch.cat([reference_unet.conv_in.weight, ref_expand_weights], dim=1) * (2/3)

            input_conv = nn.Conv2d(
                12, 320, kernel_size=3, padding=(1, 1)
            ) 

            input_conv.weight = torch.nn.Parameter(inflated_ref_in_weight)
            input_conv.bias = reference_unet.conv_in.bias 

            reference_unet.conv_in = input_conv
                
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path_scratch,
    ).to(dtype=weight_dtype, device="cuda")
        
    # image_enc = CLIPVisionModelWithProjection.from_pretrained(
    #     cfg.image_encoder_path,
    # ).to(dtype=weight_dtype, device="cuda")

    clip_preprocessor = CLIPImageProcessor()
    
    # depth_model = torch.hub.load("./ZoeDepth", "ZoeD_N", source="local", pretrained=True).to(device="cuda")
    # depth_model_path = "extern/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # depth_model_path = "/mnt/genwarp_training_checkpoint/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # depth_model = load_model(depth_model_path, 'cuda')

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    # depth_model.requires_grad_(False)

    # Explictly declare training models
    use_lora = cfg.use_lora

    if use_lora == False:

        denoising_unet.requires_grad_(True)
        reference_unet.requires_grad_(True)

        for name, param in denoising_unet.named_parameters():
            if "zero_conv" in name:
                if "up_blocks.2" in name or "up_blocks.3" in name: # or "down_blocks.0" in name or "down_blocks.1" in name:
                    print(f"!! DEBUG !! freeze {name}")
                    param.requires_grad_(False)

        for name, param in reference_unet.named_parameters():
            if "up_blocks.3" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    else:    
        # import pdb; pdb.set_trace()

        denoising_unet.requires_grad_(False)
        reference_unet.requires_grad_(False)

        # less = [name for name, param in denoising_unet.named_parameters() if "transformer" in name]
   
        den_lora_params, den_train_names = inject_trainable_lora(denoising_unet)  
        ref_lora_params, ref_train_names = inject_trainable_lora(reference_unet)  


    #TODO : temporarily can be disabled for debugging
    pose_guider.requires_grad_(True)

    # import pdb; pdb.set_trace()
    # attn_proc_hooker=XformersCrossAttentionHooker(True)
    # original_attn_proc=denoising_unet.attn_processors
    # denoising_unet.set_attn_processor(attn_proc_hooker)

    # import pdb; pdb.set_trace()

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
        feature_fusion_type=cfg.feature_fusion_type if hasattr(cfg, "feature_fusion_type") else "attention_masking",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        feature_fusion_type=cfg.feature_fusion_type if hasattr(cfg, "feature_fusion_type") else "attention_masking",
    )
    logger.info(f"Featue fusion type is '{cfg.feature_fusion_type}'")

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
    
    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.dataset.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    
    optimizer_load = False
    
    if optimizer_load:
        optimizer.load_state_dict(
            torch.load(join(cfg.model_path, 'optimizer.pth'),
                map_location='cpu'),
        )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    dataset = cfg.dataset.name
    num_viewpoints = cfg.dataset.num_viewpoints
    num_ref_viewpoints = cfg.dataset.num_ref
    target_idx = cfg.dataset.target_idx

    ENDPOINT_URL = 'https://storage.clova.ai'

    os.environ['AWS_ACCESS_KEY_ID'] = "AUIVA2ODFS9S2YDD0A75"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "VDIVVqIC9FCC0GmOQ2nNy3o7NjkWVqC4oTDOz3mM"
    os.environ['S3_ENDPOINT_URL'] = ENDPOINT_URL

    num_list = torch.arange(0, 1200)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if dataset == "co3d_duster":
        # co3d_path = "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/co3d_partial"

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(size=512),
        #     transforms.CenterCrop(size=(512,512))  # Example cropping to 224x224
        # ])

        # train_dataset = Co3DDataset(co3d_path=co3d_path, num_random_samples=num_viewpoints, transform=transform)

        # Create the DataLoader for batching
        # train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=cfg.dataset.train_bs, shuffle=True)

        # urls = [f's3://generation-research/co3d_dust3r/train/co3d_{num:06}.tar' for num in range(2700)]

        # add awscli command to urls
        # urls = [f'pipe:aws --endpoint-url={ENDPOINT_URL} s3 cp {url} -' for url in urls]
        
        co3d_directory = "/media/dataset/co3d"
        realestate_directory = "/media/dataset/Realestate"
        
        urls_1 = []
        for root, _, files in os.walk(co3d_directory):
            for file in files:
                if file.endswith(".tar"):
                    urls_1.append(os.path.join(root, file))
                
        dataset_length = 100
        epoch = 1000
        shardshuffle=True

        postprocess_fn_1 = partial(postprocess_co3d, num_viewpoints=num_ref_viewpoints)

        train_dataset = (
                wds.WebDataset(urls_1, 
                               resampled=True,
                               shardshuffle=shardshuffle, 
                               nodesplitter=wds.split_by_node,
                               workersplitter=wds.split_by_worker,
                               handler=wds.ignore_and_continue)
                .decode("pil")
                .map(postprocess_fn_1)
                .with_length(dataset_length)
                .with_epoch(cfg.dataset.train_bs * epoch)
        )

        train_dataloader = DataLoader(train_dataset, num_workers=world_size, batch_size=cfg.dataset.train_bs, persistent_workers=True)
        
        urls_2 = []
        for root, _, files in os.walk(realestate_directory):
            for file in files:
                if file.endswith(".tar"):
                    urls_2.append(os.path.join(root, file))
                        
        dataset_length = 100
        epoch = 1000
        shardshuffle=True

        postprocess_fn_2 = partial(postprocess_realestate, num_viewpoints=num_ref_viewpoints)

        train_dataset_2 = (
                wds.WebDataset(urls_2, 
                               resampled=True,
                               shardshuffle=shardshuffle, 
                               nodesplitter=wds.split_by_node,
                               workersplitter=wds.split_by_worker,
                               handler=wds.ignore_and_continue)
                .decode("pil")
                .map(postprocess_fn_2)
                .with_length(dataset_length)
                .with_epoch(cfg.dataset.train_bs * epoch)
        )

        train_dataloader_2 = DataLoader(train_dataset_2, num_workers=world_size, batch_size=cfg.dataset.train_bs, persistent_workers=True)

        # val_urls = [f's3://generation-research/co3d_dust3r/train/co3d_{num:06}.tar' for num in range(1000)]
        # val_urls = [f'pipe:aws --endpoint-url={ENDPOINT_URL} s3 cp {url} -' for url in val_urls]

        # val_dataset = (
        #         wds.WebDataset(val_urls, 
        #                        resampled=True,
        #                        shardshuffle=shardshuffle, 
        #                        nodesplitter=wds.split_by_node,
        #                        workersplitter=wds.split_by_worker,
        #                        handler=wds.ignore_and_continue)
        #         .decode("pil")
        #         .map(postprocess_fn)
        #         .with_length(dataset_length)
        #         .with_epoch(cfg.dataset.train_bs * epoch)
        # )

        # val_dataloader = DataLoader(val_dataset, num_workers=world_size, batch_size=cfg.dataset.train_bs, persistent_workers=True)

    else:
        urls = [f's3://generation-research/realestate_duster/realestate_{num:06}.tar' for num in num_list]

        # add awscli command to urls
        urls = [f'pipe:aws --endpoint-url={ENDPOINT_URL} s3 cp {url} -' for url in urls]

        dataset_length = 1200
        epoch = 10000
        shardshuffle=True

        postprocess_fn = partial(postprocess_func, num_viewpoints=num_ref_viewpoints)

        train_dataset = (
                wds.WebDataset(urls, 
                               resampled=True,
                               shardshuffle=shardshuffle, 
                               nodesplitter=wds.split_by_node,
                               workersplitter=wds.split_by_worker,
                               handler=wds.ignore_and_continue)
                .decode("pil")
                .map(postprocess_fn)
                .with_length(dataset_length)
                .with_epoch(cfg.dataset.train_bs * epoch)
        )

        train_dataloader = DataLoader(train_dataset, num_workers=world_size, batch_size=cfg.dataset.train_bs, persistent_workers=True)

    # Prepare everything with our `accelerator`.
    if not cfg.multi_dataset:
        (
            net,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            net,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        
    else:
        (
            net,
            optimizer,
            train_dataloader,
            train_dataloader_2,
            lr_scheduler,
        ) = accelerator.prepare(
            net,
            optimizer,
            train_dataloader,
            train_dataloader_2,
            lr_scheduler,
        )
    # import pdb; pdb.set_trace()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            project_name="animate-nvs",
            config=OmegaConf.to_container(cfg),
        )

    # import pdb; pdb.set_trace()

    # Train!
    total_batch_size = (
        cfg.dataset.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dataset.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            raise ValueError("Do not support latest checkpoint currently")
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    embedder, out_dim = get_embedder(2)

    if cfg.gt_cor_reg:
        # import pdb; pdb.set_trace()
        attn_proc_hooker=XformersCrossAttentionHooker(True,num_ref_views=num_ref_viewpoints)
        # original_attn_proc=denoising_unet.attn_processors
        denoising_unet.set_attn_processor(attn_proc_hooker)

    # accelerator.unwrap_model(denoising_unet).set_attn_processor(attn_proc_hooker)

    # import pdb; pdb.set_trace()

    # re10k_iter = iter(train_dataloader_2)
    # mega_iter = iter(train_dataloader_3)

    depth_list = []
    
    if cfg.embed_pointmap_norm:
        # ptsmap_min = torch.tensor([-0.4, -0.7, 0.01]).to(image_enc.device)
        # ptsmap_max = torch.tensor([0.4, 0.3, 1.5]).to(image_enc.device)
                
        ptsmap_min = torch.tensor([-0.1798, -0.2254,  0.0593]).to(image_enc.device)
        ptsmap_max = torch.tensor([0.1899, 0.0836, 0.7480]).to(image_enc.device)
        
        # real_ptsmap_min = torch.tensor([-0.1889, -0.1935,  0.0543]).to(image_enc.device)
        # real_ptsmap_max = torch.tensor([0.2030, 0.1136, 0.6233]).to(image_enc.device)
        
        pts_norm_func = PointmapNormalizer(ptsmap_min, ptsmap_max, k=0.9)
    
    if cfg.multi_dataset:
        re10k_iter = iter(train_dataloader_2)
        

    for epoch in range(first_epoch, num_train_epochs):

        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            
            if cfg.multi_dataset:
                data_iden = torch.rand(1)
                                
                if data_iden > 0.4:
                    try:
                        batch = next(re10k_iter)
                    except StopIteration:
                        re10k_iter = iter(train_dataloader_2)
                        batch = next(re10k_iter)
                else:
                    pass

            try:
                dummy = batch["image"]
            except:
                continue

            with accelerator.accumulate(net):
                # Convert videos to latent space

                switcher_stat = None

                # if cfg.switcher:
                    

                    # if sw > 0.5:
                    #     switcher_stat = "image"
                    # else:
                    #     switcher_stat = "geometry"
                        
                view_rand = torch.rand(1)
                if view_rand > 0.6:
                    target_idx = random.sample(range(-1,1),1)[0]
                else:
                    target_idx = random.sample(range(1,cfg.dataset.num_viewpoints-1),1)[0]
                    
                src_idx = [k for k in range(num_viewpoints) if k is not target_idx]
                                
                images = dict(ref=[batch["image"][:,k].unsqueeze(1) for k in src_idx], tgt=batch["image"][:,target_idx])

                with torch.no_grad():
                    if dataset == "co3d":
                        img_size = batch["orig_img_size"][0,0]

                        ref_camera = dict(rot=batch["R"][:,src_idx],
                                        trans=batch["T"][:,src_idx] ,
                                        intrinsic=batch["K"][:,src_idx],
                                        focals=batch["focal_length"][:,src_idx],
                                        orig_img_size=batch["orig_img_size"][:,src_idx])       

                        target_idx = target_idx

                        tgt_camera = dict(rot=batch["R"][:,target_idx],
                                        trans=batch["T"][:,target_idx] ,
                                        intrinsic=batch["K"][:,target_idx],
                                        focals=batch["focal_length"][:,target_idx],
                                        orig_img_size=batch["orig_img_size"][:,target_idx])
                        
                        camera_info = dict(ref=ref_camera, tgt=tgt_camera)

                        mask_depth = False
                        depth =  batch["depth"][:,src_idx].requires_grad_(False)

                        if mask_depth:
                            depth = batch["mask"][:,src_idx] * depth

                        points = pose_to_ray(ref_camera, depth, img_size, num_viewpoints=num_ref_viewpoints).float()
                        correspondence = dict(ref=points, tgt=None)
                    
                    elif dataset == "realestate" or dataset == "co3d_duster":
                        device = batch['pose'].device
                        batch_size = batch['pose'].shape[0]

                        ref_camera=dict(pose=batch['pose'][:,src_idx].float(), 
                            focals=batch['focals'][:,src_idx], 
                            orig_img_size=torch.tensor([512, 512]).to(device))
    
                        tgt_camera=dict(pose=batch['pose'][:,target_idx].float(),
                            focals=batch['focals'][:, target_idx],
                            orig_img_size=torch.tensor([512, 512]).to(device))

                        camera_info = dict(ref=ref_camera, tgt=tgt_camera)
                        correspondence = dict(ref=batch['points'][:, src_idx].float(), tgt=batch['points'][:, target_idx].float())
                    
                    tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth = mari_embedding_prep(cfg, images, correspondence, ref_camera, tgt_camera, src_idx, batch_size, device)
                    src_images = images["ref"]
                    
                    # ref_reshaped_depth = ref_depth.reshape(batch_size, num_ref_viewpoints, 3, 512, 512)
                    # src_depths = [ref_reshaped_depth[:,i].unsqueeze(1) for i in range(num_ref_viewpoints)]
                    
                    args = dict(
                        src_images=src_images,
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
                        args["mesh_pts"] = mesh_pts
                        args["mesh_depth"] = mesh_depth
                    
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
                                        
                    if cfg.embed_pointmap_norm:
                        args["pts_norm_func"] = pts_norm_func
                    
                    # Embedding preparation
                    if dataset == "co3d":
                        conditions, renders = prepare_co3d_embedding(**args)
                    
                    elif dataset == "realestate" or dataset == "co3d_duster":
                        conditions, renders = prepare_duster_embedding(**args)
                        
                    # save_image(images["tgt"],"tgt_image.png")
                    # save_image(renders["warped"], "tgt_warped.png")
                    # save_image(tgt_depth_norm * 0.5 + 0.5,"tgt_depth.png") 
                    # import pdb; pdb.set_trace()
                    
                    if cfg.geo_setting == "depth":
                        latents = vae.encode(tgt_depth_norm.to(weight_dtype)).latent_dist.sample()
                        
                    elif cfg.geo_setting == "pointmap":
                        ref_pointmaps = correspondence["ref"].reshape(-1,512,512,3).permute(0,3,1,2)
                        tgt_pointmap = correspondence["tgt"].permute(0,3,1,2)
                        
                        if cfg.conditioning_pointmap_norm:
                            minmax_set = True  
                            
                            if minmax_set:
                                ptsmap_min = torch.tensor([-0.1798, -0.2254,  0.0593]).to(image_enc.device)
                                ptsmap_max = torch.tensor([0.1899, 0.0836, 0.7480]).to(image_enc.device)
                                
                            else:                      
                                ptsmap_min = ref_pointmaps.permute(0,2,3,1).reshape(-1,3).min(dim=0)[0]
                                ptsmap_max = ref_pointmaps.permute(0,2,3,1).reshape(-1,3).max(dim=0)[0]
                            
                            ref_pointmaps = torch.clip((ref_pointmaps - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                            tgt_pointmap = torch.clip((tgt_pointmap - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                            
                        if cfg.use_warped_cond:
                            if cfg.use_mesh:
                                warped_tgt_pointmap = mesh_pts.permute(0,3,1,2)
                            else:
                                print("Warped point cloud not implemented")
                            
                            if cfg.conditioning_pointmap_norm:
                                warped_tgt_pointmap = torch.clip((warped_tgt_pointmap - ptsmap_min[None,...,None,None]) / (ptsmap_max[None,...,None,None] - ptsmap_min[None,...,None,None]) * 2 - 1.0, min=-1.0, max=1.0)
                                
                        latents = vae.encode(tgt_pointmap.to(weight_dtype)).latent_dist.sample()

                    if cfg.multitask:
                        if cfg.multitask_mode == "depth":
                            encoded_geo = encode_depth(tgt_depth, vae, weight_dtype)
                            if not cfg.switcher:
                                latents = torch.cat((latents, encoded_geo), dim=1)

                        elif cfg.multitask_mode == "loc":
                            pts_locs = correspondence["ref"].reshape(1,-1,3)
                            max = torch.max(pts_locs, dim=-2)[0]
                            min = torch.min(pts_locs, dim=-2)[0]
                            tgt_corr = ((correspondence["tgt"] - min[:,None,None,:]) / (max[:,None,None,:] - min[:,None,None,:])) * 2 - 1
                            encoded_geo = vae.encode(tgt_corr.permute(0,3,1,2)).latent_dist.sample()

                            if not cfg.switcher:
                                latents = torch.cat((latents, encoded_geo), dim=1)

                        elif cfg.multitask_mode == "normal":
                            pass

                        if cfg.switcher:
                            encoded_geo = encoded_geo.unsqueeze(2) 
                            encoded_geo = encoded_geo * 0.18215  
                    
                        if cfg.use_ref_expand:
                            ref_input = conditions["ref_correspondence"]
                            ref_encoded_batches = []
                            for ref_corrs in ref_input:
                                ref_encoded_batches.append(vae.encode(ref_corrs.permute(0,3,1,2)).latent_dist.sample()[None,...])
                            ref_corr_latents = torch.cat(ref_encoded_batches, dim=0).permute(1,0,2,3,4) 
                            
                    # import pdb; pdb.set_trace()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215   
                    
                    # Conditioning with partially projected image
                    warped_image = renders["warped"]
                    warped_image = warped_image * mesh_normal_mask.permute(0,3,1,2)
                    # if step % 4 == 0:
                    #     save_image(torch.cat((warped_image, images["tgt"]),dim=0),"check.png")
                    warped_latents = vae.encode(warped_image).latent_dist.sample().unsqueeze(2)  
                    warped_latents = warped_latents *  0.18215              
                    
                    if cfg.use_warped_cond:
                        if cfg.geo_setting == "depth":
                            warped_depth = mesh_depth.unsqueeze(1).repeat(1,3,1,1)
                            norm_warped_depth = depth_normalize(cfg, warped_depth)
                            
                            clip = True
                            
                            if clip:
                                min_val = -1.0
                                max_val = 1.0
                                norm_warped_depth = torch.clip(norm_warped_depth, min=min_val, max=max_val)
                                
                                warp_depth_latents = vae.encode(norm_warped_depth).latent_dist.sample().unsqueeze(2)  
                                warp_depth_latents = warp_depth_latents * 0.18215
                                
                        if cfg.geo_setting == "pointmap":
                            warp_depth_latents = vae.encode(warped_tgt_pointmap).latent_dist.sample().unsqueeze(2) 
                            warp_depth_latents = warp_depth_latents * 0.18215 
                    
                        # mask = ( warped_deph != 0.)
                            
                        # save_image(norm_warped_depth * 0.5 + 0.5,"warped_depth.png")
                        # save_image((tgt_depth_norm * 0.5 + 0.5) * mask,"w_tgt_depth.png")
                        
                        # tee = (tgt_depth_norm * 0.5 + 0.5) * mask
                        # nee = norm_warped_depth * 0.5 + 0.5
                        
                        # import pdb; pdb.set_trace()
                    
                is_warped_feat_injection = cfg.feature_fusion_type == 'warped_feature'
                   
                noise = torch.randn_like(latents)

                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                
                if not cfg.switcher:
                    timesteps = torch.randint(
                        0,
                        train_noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()
                else:
                    # Lotus
                    timesteps = 999 * torch.ones(bsz).to(latents.device).long()

                # additional_cnd = torch.cat([embed, torch.zeros_like(mask.unsqueeze(1))], dim=1).unsqueeze(2).to(weight_dtype)     
                # additional_cnd_2 = torch.cat([warped_image, mask.unsqueeze(1)], dim=1).unsqueeze(2)  # (bs, 2, *1*, 512, 512)
                
                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_depth_list = []
                ref_image_list = []

                ref_stack = torch.cat(images["ref"], dim=1)
                B, V, C, H, W = ref_stack.shape

                ref_stack = ref_stack.reshape(-1,C,H,W) 
                               
                for batch_idx, (ref_d, ref_img, clip_img) in enumerate(
                    zip(
                        ref_depth if cfg.geo_setting == "depth" else ref_pointmaps,
                        ref_stack,
                        clip_preprocessor(ref_stack*0.5+0.5, do_rescale=False, return_tensors="pt").pixel_values,
                    )
                ):  
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                        
                    ref_depth_list.append(ref_d)
                    ref_image_list.append(ref_img)
                
                with torch.no_grad():
                    ref_depth_stack = torch.stack(ref_depth_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_depth_latents = vae.encode(
                        ref_depth_stack
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_depth_latents = ref_depth_latents * 0.18215
                    
                    if cfg.use_ref_cond_masking:
                        # import pdb; pdb.set_trace()
                        mask = ( torch.rand_like(ref_depth_stack) >= 0.8).float()
                        
                        ref_masked_depth_stack = mask * ref_depth_stack 
                        
                        ref_masked_depth_latents = vae.encode(
                            ref_masked_depth_stack
                        ).latent_dist.sample() 
                        
                        ref_masked_depth_latents = ref_masked_depth_latents * 0.18215
                    
                    ref_img_stack = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img_stack
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215
                    
                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                if not cfg.switcher:
                    task_emb = None
                    noisy_latents = train_noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                elif cfg.switcher:
                    # This setting means that image is noised with geo as condition
                    # if switcher_stat == "image":
                    #     noisy_latents = torch.cat((noisy_latents, encoded_geo), dim=1)
                    #     task_emb = torch.tensor([0, 1]).float().unsqueeze(0).repeat(1, 1).to(noisy_latents.device)
                    # elif switcher_stat == "geometry"

                    img_latents = latents 
                    latents = train_noise_scheduler.add_noise(encoded_geo, noise, timesteps)
                    # noisy_latents = torch.cat((latents, img_latents), dim=1)
                    noisy_latents = latents

                    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(noisy_latents.device)
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
                    # elif channel_length == 8:
                    #     task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 2)

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                                
                image_prompt_embeds = image_prompt_embeds.reshape(B,V,1,-1).permute(1,0,2,3)
                ref_image_latents = ref_image_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
                ref_depth_latents = ref_depth_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
                
                ref_latents = torch.cat((ref_image_latents, ref_depth_latents), dim=2)
                noisy_latents = torch.cat((warped_latents, noisy_latents), dim=1)
                
                if cfg.use_warped_cond:
                    noisy_latents = torch.cat((noisy_latents, warp_depth_latents), dim=1)
                
                if cfg.use_ref_expand:
                    if not cfg.use_ref_cond_masking:
                        ref_latents = torch.cat((ref_latents, ref_depth_latents), dim=2)
                    else:
                        ref_masked_depth_latents = ref_masked_depth_latents.reshape(B,V,-1,64,64).permute(1,0,2,3,4)
                        ref_latents = torch.cat((ref_latents, ref_masked_depth_latents), dim=2)
                        # import pdb; pdb.set_trace()
                     
                model_pred = net(
                    noisy_latents.to(weight_dtype),
                    timesteps,
                    ref_latents.to(weight_dtype),
                    image_prompt_embeds.to(weight_dtype),
                    conditions["ref_embeds"], # V List of (B, 16, 512, 512) latents
                    conditions["tgt_embed"].to(weight_dtype), # Tensor of (B, 16, 512, 512)
                    correspondence=None,
                    weight_dtype=weight_dtype,
                    gt_target_coord_embed = conditions['gt_tgt_embed'] if conditions['gt_tgt_embed'] != None else None,
                    task_emb=task_emb
                    # else {'depth':src_depth[:,:1,...], 'R':R, 'T':T, 'K':K, 'H':cfg.dataset.resolution_h, 'W':cfg.dataset.resolution_w},
                )
                
                if cfg.gt_cor_reg:
                    m_loss = attn_proc_hooker.mean_loss
                    v_loss = attn_proc_hooker.var_loss

                    attn_proc_hooker.clear()

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1

                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )

                    if cfg.multitask_weighting:
                        image_loss = loss[:,:4]
                        multi_loss = loss[:,4:]
                        loss = (
                            ((cfg.image_weight * image_loss.mean(dim=list(range(1, len(loss.shape))))) + ((1 - cfg.image_weight) * multi_loss.mean(dim=list(range(1, len(loss.shape))))))
                            * mse_loss_weights
                        )

                    else:
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )

                    loss = loss.mean()
                
                raw_loss = loss
                
                if cfg.gt_cor_reg:
                    reg_loss = 1000 * (m_loss + v_loss)
                    loss += reg_loss

                # if torch.isnan(loss):
                #     save_image(torch.cat((ref_stack,images["tgt"])),f"Nan_{epoch}_{step}.png")
                #     import pdb; pdb.set_trace()
                    # loss = 0. * loss
                    # import pdb; pdb.set_trace()
                    
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.dataset.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )

                # import pdb; pdb.set_trace()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
 
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                if cfg.gt_cor_reg:
                    accelerator.log({"raw_loss": raw_loss}, step=global_step)
                    accelerator.log({"reg_loss": reg_loss}, step=global_step)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)
                        
                        unwrap_net = accelerator.unwrap_model(net)
                        save_checkpoint(
                            unwrap_net.reference_unet,
                            save_dir,
                            "reference_unet",
                            global_step,
                            total_limit=1,
                        )
                        save_checkpoint(
                            unwrap_net.denoising_unet,
                            save_dir,
                            "denoising_unet",
                            global_step,
                            total_limit=1,
                        )
                        save_checkpoint(
                            unwrap_net.pose_guider,
                            save_dir,
                            "pose_guider",
                            global_step,
                            total_limit=1,
                        )

                # if global_step % cfg.val.validation_steps == 0:
                # with torch.no_grad():
                #     psnr_mean, imgs = log_validation(net, accelerator, ref_reader=reference_control_reader, ref_writer=reference_control_writer, multitask = cfg.multitask, depth_condition=cfg.use_depthmap, use_mesh=cfg.use_mesh, use_normal=cfg.use_normal)
                #     accelerator.log({"psnr": psnr_mean}, step=global_step)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (epoch + 1) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=3,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=3,
            )
            save_checkpoint(
                unwrap_net.pose_guider,
                save_dir,
                "pose_guider",
                global_step,
                total_limit=3,
            )

        # if global_step % cfg.val.validation_steps == 0:


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/train_realestate.yaml")
    parser.add_argument("--config", type=str, default="././train_configs/marigold_train_co3d.yaml")
    # parser.add_argument("--config", type=str, default="/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/train_configs/train_co3d_depth.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)