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

import open3d as o3d
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
 
def batchify_known(run_setting):
    if run_setting == "real_known":
    
        image_dir_0 = "/media/dataset/train/fff5c6b9222f5afe"
        image_dir_1 = "/media/dataset/train/fff7d26426284b52"
        image_dir_2 = "/media/dataset/train/fff1040caa0c235c"
        image_dir_3 = "/media/dataset/train/fffda70800952a46"

        image_dirs = [image_dir_0, image_dir_1, image_dir_2, image_dir_3]

        fx = [dir.split("/")[-1] for dir in image_dirs]

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
        image_dir_0 = "/media/co3d/dataset/car/106_12658_23657/images"
        image_dir_1 = "/media/co3d/dataset/car/194_20901_41098/images"
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
    
    all_data=[]

    for frame_idx, frame_imgs in enumerate(frames):            
        images = []
        src_images = []
        num_frames = 10
        p = fx[frame_idx]

        img_idx = torch.floor(torch.linspace(0, frame_imgs.shape[0]-50, num_frames+2)).int()

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
            pts3d = points_for_saving
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
            src_idx = [1, -1]
        elif run_setting == "co3d_known":
            src_idx = [0, 3, -3, -1]

        t_idx = [i for i in range(len(img_idx)) if i not in src_idx]
        alphas = poses[t_idx]
        
        try:
            pseudo_pts_tensor = torch.load(f"{p}_pseudo_pts_tensor.pt")
        except:
            pass
        
        image_list = []
        cor_list = []
        ref_cam_list = []
        tgt_cam_list = []
        cam_info_list = []

        for idxc, alpha in tqdm(enumerate(alphas)):

            new_target_pose = alpha
            target_idx = t_idx[idxc]
            
            # Preparing input information for GenWarp
            images = dict(ref=[src_images[k].to(device)[None,...] for k in src_idx], tgt=src_images[target_idx].to(device))
            correspondence = dict(ref=torch.cat([pts3d[k][None,...] for k in src_idx])[None,...].to(device), tgt=pts3d[target_idx][None,...].to(device))

            ref_camera=dict(pose=poses[src_idx].to(device)[None,...], 
                            focals=focals[src_idx].to(device)[None,...], 
                            orig_img_size=torch.tensor([512, 512]).to(device))
        
            tgt_camera=dict(pose=new_target_pose[None,...].to(device),
                            focals=focals[target_idx][None,...].to(device),
                            orig_img_size=torch.tensor([512, 512]).to(device))
            
            camera_info = dict(ref=ref_camera, tgt=tgt_camera)
            
            image_list.append(images)
            cor_list.append(correspondence)
            ref_cam_list.append(ref_camera)
            tgt_cam_list.append(tgt_camera)
            cam_info_list.append(camera_info)
        
        info_dict = dict(
            key=fx[frame_idx],
            image_list = image_list,
            cor_list = cor_list,
            ref_cam_list = ref_cam_list,
            tgt_cam_list = tgt_cam_list,
            cam_info_list = cam_info_list
        )

        all_data.append(info_dict)       
        
    return all_data      
            
            