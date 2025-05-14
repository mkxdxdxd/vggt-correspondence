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


from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

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



if __name__ == "__main__":

    # genwarp_cfg = dict(
    #     pretrained_model_path='./checkpoints',
    #     checkpoint_name='multi1',
    #     half_precision_weights=False
    # )

    alpha = dict(checkpoint_name = 'marigold_every',
        depth_condition = True,
        use_mesh = True,
        use_normal = True,
        use_plucker = False,
        use_opt=False,
        use_full_gt=False,
        multitask=False,
        ref_expand=False) 
    
    # alpha = dict(checkpoint_name = 'co3d_attn_reg_next',
    #     depth_condition = True,
    #     use_mesh = False,
    #     use_normal = False,
    #     use_plucker = False,
    #     use_full_gt= False,
    #     use_opt= False)

    # beta = dict(checkpoint_name = 'co3d_attn_reg',
    #     depth_condition = True,
    #     use_mesh = False,
    #     use_normal = False,
    #     use_plucker = False,
    #     use_full_gt= False,
        # use_opt= False)
    
    # gamma = dict(checkpoint_name = 'co3d_norm_mask',
    #     depth_condition = True,
    #     use_mesh = True,
    #     use_normal = True,
    #     use_plucker = False,
    #     use_opt=False,
    #     use_full_gt=False,
    #     multitask=False,
    #     ref_expand=False)  
    
    # gamma = dict(checkpoint_name = 'multi_best',
    #     depth_condition = False,
    #     use_mesh = False,
    #     use_normal = False,
    #     use_plucker = False,
    #     use_opt=False,
    #     use_full_gt=False,
    #     multitask=False,
    #     ref_expand=False) 

    # gamma = dict(checkpoint_name = 'co3d_multitask_loc_weighted_09_ref_input',
    #     depth_condition = True,
    #     use_mesh = True,
    #     use_normal = True,
    #     use_plucker = False,
    #     use_opt=False,
    #     use_full_gt=False,
    #     multitask=True,
    #     ref_expand=True)    

    # sigma = dict(checkpoint_name = 'co3d_attn_reg_next',
    #     depth_condition = True,
    #     use_mesh = False,
    #     use_normal = False,
    #     use_plucker = False,
    #     use_full_gt= False,
    #     use_opt= True)
    
    # sigma = dict(checkpoint_name = 'co3d_four_depth_gt_opt',
    #     depth_condition = True,
    #     use_mesh = False,
    #     use_normal = False,
    #     use_plucker = False,
    #     use_opt=True,
    #     use_full_gt=False)    

    # kima = dict(checkpoint_name = 'co3d_four_depth_gt_full',
    #     depth_condition = True,
    #     use_mesh = False,
    #     use_normal = False,
    #     use_plucker = False,
    #     use_full_gt=True,
    #     use_opt=False)  

    # settings = [alpha, beta, gamma, sigma]

    settings = [alpha]
    
    for setting in settings:

        checkpoint_name = setting['checkpoint_name']
        depth_condition = setting['depth_condition']
        use_mesh = setting['use_mesh']
        use_normal = setting['use_normal']
        use_plucker = setting['use_plucker']
        use_full_gt = setting['use_full_gt']
        use_opt = setting['use_opt']
        multitask = setting["multitask"]
        
        genwarp_cfg = dict(
            # pretrained_model_path='./checkpoints',
            pretrained_model_path='/media/multiview-gen/checkpoints',
            checkpoint_name=checkpoint_name,
            half_precision_weights=False,
            embedder_input_dim=3,
            depth_condition=depth_condition,
            use_mesh=use_mesh,
            use_normal=use_normal,
            use_plucker=use_plucker,
            multitask=multitask,
            ref_expand=setting["ref_expand"]
        )

        genwarp_nvs = GenWarp(cfg=genwarp_cfg)

        ENDPOINT_URL = 'https://storage.clova.ai'

        os.environ['AWS_ACCESS_KEY_ID'] = "AUIVA2ODFS9S2YDD0A75"
        os.environ['AWS_SECRET_ACCESS_KEY'] = "VDIVVqIC9FCC0GmOQ2nNy3o7NjkWVqC4oTDOz3mM"
        os.environ['S3_ENDPOINT_URL'] = ENDPOINT_URL

        run_setting = "co3d_known"
        # run_setting = "real_known"

        if run_setting == "real_unknown":

            urls = [f's3://generation-research/realestate/realestate_{num:06}.tar' for num in range(2300,2413)]

            # add awscli command to urls
            urls = [f'pipe:aws --endpoint-url={ENDPOINT_URL} s3 cp {url} -' for url in urls]

            dataset_length = 1200
            epoch = 10000
            shardshuffle=False
            world_size=1

            train_dataset = (
                    wds.WebDataset(urls, 
                                    resampled=True,
                                    shardshuffle=shardshuffle, 
                                    # nodesplitter=wds.split_by_node,
                                    # workersplitter=wds.split_by_worker,
                                    handler=wds.ignore_and_continue)
                    .decode("pil")
                    .map(postprocess_fn)
                    .with_length(dataset_length)
            )

            train_dataloader = DataLoader(train_dataset, num_workers=world_size, batch_size=1, persistent_workers=True)

            for sample in train_dataloader:

                frame_imgs = sample["img"][0]
                frame_poses = sample["pose"][0].float()
                frame_poses = torch.cat((frame_poses, torch.tensor([0,0,0,1]).to(frame_poses.device)[None,None,...].repeat(frame_poses.shape[0],1,1)), dim=1)

                if frame_imgs.shape[0] > 50:
                    break

        # HERE

        elif run_setting == "real_known":
        
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
                    transforms.CenterCrop(512),
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

            # torch.save(points_for_saving,f"real_duster_results/scene_{p}_points.pt")
            # torch.save(poses,f"real_duster_results/scene_{p}_poses.pt")
            # torch.save(focals,f"real_duster_results/scene_{p}_focals.pt")
            
            else:
                pts3d = torch.load(f"duster_results/scene_{p}_points.pt", map_location="cpu")
                poses = torch.load(f"duster_results/scene_{p}_poses.pt", map_location="cpu")
                focals = torch.load(f"duster_results/scene_{p}_focals.pt", map_location="cpu")
                new_pose=False

            if run_setting == "real_known":
                alphas = poses[1:-1]
            elif run_setting == "co3d_known":
                src_idx = [0, 3, 6, -1]

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

                    images = dict(ref=[src_images[k].to(device) for k in src_idx], tgt=src_images[target_idx].to(device))
                    correspondence = dict(ref=torch.cat([pts3d[k][None,...].to(device) for k in src_idx]), tgt=pts3d[target_idx])

                    ref_camera=dict(pose=poses[src_idx].to(device), 
                                    focals=focals[src_idx].to(device), 
                                    orig_img_size=torch.tensor([512, 512]).to(device))
                
                    tgt_camera=dict(pose=new_target_pose[None,...].to(device),
                                    focals=focals[target_idx][None,...].to(device),
                                    orig_img_size=torch.tensor([512, 512]).to(device))
                    
                    camera_info = dict(ref=ref_camera, tgt=tgt_camera)

                    # print(correspondence["ref"].shape)

                    # import pdb; pdb.set_trace()

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
                        mesh_normal_mask = []

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

                        normal_mask = False

                        if normal_mask:
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

                            if normal_mask:
                                mesh_normal_mask = (1 - torch.stack(mesh_normal_mask).float()).to(device)

                                mesh_pts = mesh_pts.to(device) * mesh_normal_mask
                                mesh_depth = mesh_depth.to(device) * mesh_normal_mask[...,0]
                                mesh_normals = mesh_normals.to(device) * mesh_normal_mask

                            # save_image(mesh_depth[0],"new.png")
                            # save_image(images["tgt"][0], "neww.png")
                            # save_image(mesh_normals.permute(0,3,1,2), "newww.png")
                            # save_image(mesh_ref_normals[0],"plz.png")
                            # save_image(torch.cat((images["ref"][0][0],images["ref"][1][0])),"plzz.png")

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
                        dataset = "realestate",
                        masking_percent=masking_percent,
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
                    
                    if use_full_gt:
                        args["use_full_gt"] = True
                    
                    elif use_opt:
                        args["use_opt"] = True


                    renders = genwarp_nvs(
                        **args
                    )

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

                    z_mask = (renders['tgt_depth'] == 0).float() * 100
                    # import pdb; pdb.set_trace()

                    # depth_var = compute_depth_variance(torch.clamp(renders['tgt_depth'] * 3, max=1), 10)[:512,:512].permute(2,0,1)

                    maxpool_depth = compute_max_within_patch(renders['tgt_depth'] + z_mask)[:512,:512].permute(2,0,1)

                    # depth_var = compute_depth_variance(renders['tgt_depth'], 3)[:512,:512].permute(2,0,1)

                    # renders['tgt_depth']

                    # import pdb; pdb.set_trace()

                    # save_image((depth_var > 0.007).float(),"helloo.png")

                    if depth_condition:
                        # import pdb; pdb.set_trace()
                        depth_img = apply_heatmap(torch.clamp(renders['tgt_depth'] * 3, max=1))
                        depth.append(depth_img[0])
                        # depth_var = apply_heatmap(torch.clamp(depth_var[None,...], max=1))
                        max_depth = apply_heatmap(torch.clamp(maxpool_depth[None,...] * 3, max=1))
                        
                    if use_mesh:
                        depth_img = apply_heatmap(torch.clamp(mesh_depth, max=1))

                    space = torch.ones(3,512,80).to(device)

                    # import pdb; pdb.set_trace()

                    if not use_mesh:
                        if not depth_condition:
                            stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1)
                        else:
                            stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, depth_img[0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1)
                            # stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['correspondence'][0].to(device), space, depth_img[0].to(device), space, depth_var[0].to(device), space, max_depth[0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1)
                    else:
                        # import pdb; pdb.set_trace()
                        stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['warped'][0].to(device), space, depth_img[0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device)), dim=-1)

                    ult.append(stack_img)

                    if renders["other"] is not None:

                        other_diff = torch.sqrt((renders["other"].to(device) - renders["correspondence"].to(device))**2).sum(dim=1).unsqueeze(1)
                        other_diff = apply_heatmap(other_diff)
                        oth_stack_img = torch.cat((src_images[target_idx][0].to(device), space, renders['synthesized'][0].to(device), space, diff_gt[0].to(device), space, renders["correspondence"][0].to(device), space, renders["other"][0].to(device), space, other_diff[0].to(device)), dim=-1)

                        other_ult.append(oth_stack_img)


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

                if len(other_ult) >= 1:
                    make_video(other_ult, now, folder_name="multi_results")

                # make_video(diffmap_1, now, folder_name="diff_1")
                # make_video(diffmap_2, now, folder_name=f"diff_2_{masking_percent}")
                save_image(torch.cat(images["ref"]), image_dir)
                print(psnrs)