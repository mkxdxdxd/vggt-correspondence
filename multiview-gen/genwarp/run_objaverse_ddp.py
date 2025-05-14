import os
import sys
import glob
import torch
import numpy as np
import imageio
import random
import json
import io

import webdataset as wds

from PIL import Image
from time import gmtime, strftime
from tqdm import tqdm

from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
# from genwarp import GenWarp

import torchvision.transforms as transforms

import torch.nn.functional as F

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import open3d as o3d
from typing import cast

# from pointcloud_loader import get_rgbd_point_cloud

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.genwarp.')))

from genwarp.dust3r.inference import inference
from genwarp.dust3r.model import AsymmetricCroCo3DStereo
from genwarp.dust3r.utils.image import load_images
from genwarp.dust3r.image_pairs import make_pairs
from genwarp.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from genwarp.dust3r.utils.geometry import find_reciprocal_matches, xy_grid

# from genwarp.utils import (
#     reprojector,
#     ndc_rasterizer,
#     one_to_one_rasterizer
# )


os.environ["MASTER_ADDR"] = "localhost"  # Use "localhost" for single-machine setups
os.environ["MASTER_PORT"] = "11600"      # Choose an available port

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


def save_dataset(folder_name, instance, images, focals, poses, pts3d):
    """
    Save the dataset with the specified structure, including images, camera parameters,
    and point clouds without normalization.
    
    Args:
        folder_name (str): The path to the folder where the dataset will be saved.
        images (torch.Tensor): Tensor of images with shape (12, 3, 512, 512).
        focals (list of torch.Tensor): List of focal lengths.
        poses (list of torch.Tensor): List of camera poses.
        pts3d (torch.Tensor): Point cloud tensor of shape (12, 512, 512, 3).
    """

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Initialize an empty list for annotation data
    annotation_data = []

    # Save images and point cloud maps
    for i in range(12):
        # Convert image tensor to a PIL image (assuming images are in range [0, 1] or [0, 255])
        img = images[i]
        # .permute(1, 2, 0).cpu().numpy()  # Convert to (512, 512, 3) format
        # img = (img * 255).astype(np.uint8)  # Convert to uint8 for saving as PNG
        # img_pil = Image.fromarray(img)

        # Save the image as PNG
        img_path = os.path.join(folder_name, f'image_{i+1}.png')
        save_image(img, img_path)

        # Save the point cloud map as .npy (preserving original values)
        point_cloud = pts3d[i].detach().cpu().numpy()  # Convert to numpy array (512, 512, 3)
        npy_path = os.path.join(folder_name, f'pts3d_{i+1}.npy')
        np.save(npy_path, point_cloud)

        # Add the focal length and pose to annotation data
        focal = focals[i].detach().cpu().numpy()  # Convert focal length to numpy
        pose = poses[i].detach().cpu().numpy()  # Convert pose to numpy

        # Store the data in a dictionary for later saving
        annotation_data.append({
            'focal': focal,
            'pose': pose
        })

    # Save the annotation data as a numpy file
    annotation_file_path = os.path.join(folder_name, 'annotations.npy')
    np.save(annotation_file_path, annotation_data)

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


def decode_image(data, color, has_alpha: bool = True):
    img = Image.open(io.BytesIO(data))
    img = np.array(img, dtype=np.float32)
    if has_alpha:
        img[img[:, :, -1] == 0.0] = color
    return Image.fromarray(np.uint8(img[:, :, :3]))


def make_4x4_matrix(matrix):
    """
    from 3x4 matrix to 4x4 matrix by appending [0, 0, 0, 1] to the bottom row
    """
    if isinstance(matrix, torch.Tensor):
        output = torch.zeros([4, 4], dtype=matrix.dtype)
    elif isinstance(matrix, np.ndarray):
        output = np.zeros([4, 4], dtype=matrix.dtype)
    else:
        raise TypeError(f"{type(matrix)} is unsupported")

    output[:3,:4] = matrix[:3,:4]
    output[3, 3] = 1.0
    return output


def make_intrinsic_matrix(fov_rad: float, h: int, w: int) -> torch.Tensor:
    """
    make intrinsic matrix from fov, height, width
    """
    focal_x = w / (2 * np.tan(fov_rad))
    focal_y = h / (2 * np.tan(fov_rad))

    intrinsic = torch.tensor([[focal_x, 0.0, w * 0.5, 0.0],
                               [0.0, focal_y, h * 0.5, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]],
                             dtype=torch.float32)
    return intrinsic

def compute_inverse_transform(matrix):
    """
    efficient computation of inverse transform matrix
    """
    if isinstance(matrix, torch.Tensor):
        output = torch.zeros([4, 4], dtype=matrix.dtype)
    elif isinstance(matrix, np.ndarray):
        output = np.zeros([4, 4], dtype=matrix.dtype)
    else:
        raise TypeError(f"{type(matrix)} is unsupported")

    output[..., :3, :3] = matrix[..., :3, :3].T
    output[..., :3, 3:] = - output[..., :3, :3] @ matrix[..., :3, 3:]
    output[..., 3, 3] = 1.0
    return output

def postprocess_fn(sample):

    color_background = [255.0, 255.0, 255.0, 255.0]

    total_views = 12

    fov_rad = np.deg2rad(49.1)  # for objaverse rendering dataset

    num_views_each = [5, 7]
    num_selected_views = sum(num_views_each)

    assert num_selected_views <= total_views, \
        f"num_selected_views={num_selected_views} > total_view={total_views}"

    indices = np.concatenate([np.array([0], dtype=np.int32), np.arange(total_views-1, dtype=np.int32) + 1])

    image_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(
            (512, 512), 
            interpolation=transforms.InterpolationMode.BICUBIC, 
            antialias=True
        ),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    # import pdb; pdb.set_trace()

    rgbs, intrinsics, c2ws = [], [], []

    for index in indices:
        index_str = f"{index:03d}"
        png = sample[f"png_{index_str}"]
        npy = sample[f"npy_{index_str}"]
        image = image_transform(decode_image(png, color_background))

        w2c = wds.autodecode.npy_loads(npy).astype(np.float32)
        w2c = make_4x4_matrix(torch.tensor(w2c))
        c2w = compute_inverse_transform(w2c)
        c2w[..., :3, :3] *= -1
        intrinsic = make_intrinsic_matrix(fov_rad=fov_rad, h=image.shape[1], w=image.shape[2])

        rgbs.append(image)
        intrinsics.append(intrinsic)
        c2ws.append(c2w)

    rgbs, intrinsics, c2ws = map(lambda x: torch.stack(x), (rgbs, intrinsics, c2ws))

    # rand_idx = torch.randperm(12)
    # train_idx = rand_idx[:num_views_each[0]]
    # test_idx = rand_idx[num_views_each[0]:]

    # support_rgbs = rgbs[train_idx]
    # support_intrinsics = intrinsics[train_idx]
    # support_c2ws = c2ws[train_idx]

    # query_rgbs = rgbs[test_idx]
    # query_intrinsics = intrinsics[test_idx]
    # query_c2ws = c2ws[test_idx]

    support_rgbs, query_rgbs = torch.split(rgbs, num_views_each)
    support_intrinsics, query_intrinsics = torch.split(intrinsics, num_views_each)
    support_c2ws, query_c2ws = torch.split(c2ws, num_views_each)

    # if config.use_relative:
    #     inverse_support_c2ws = torch.inverse(support_c2ws)
    #     support_c2ws = inverse_support_c2ws @ support_c2ws
    #     query_c2ws = inverse_support_c2ws @ query_c2ws

    return dict(
        instance_name=sample['__key__'],
        support_rgbs=support_rgbs,
        support_intrinsics=support_intrinsics,
        support_c2ws=support_c2ws,
        query_rgbs=query_rgbs,
        query_intrinsics=query_intrinsics,
        query_c2ws=query_c2ws,
    )

class DusterRun(nn.Module):
    def __init__(self):
        super(DusterRun, self).__init__()

        device = "cuda"
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        # self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)

    def forward(self, batch):

        images_tensor = torch.cat((batch["support_rgbs"], batch["query_rgbs"]), dim=0)
        # images has dimension (12, 3, 512, 512)

        device = images_tensor.device

        images = [dict(img = image[None,...], idx=i, instance=i, true_shape=np.array([[512,512]])) for i, image in enumerate(images_tensor) ]

        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)
        
        # view1, pred1 = output['view1'], output['pred1']
        # view2, pred2 = output['view2'], output['pred2']

        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()

        focals = [k.to(device) for k in focals]
        poses = [k.to(device) for k in poses]
        pts3d = [k.to(device) for k in pts3d]

        folder_name = batch["instance_name"]

        instance_savedir = os.path.join("/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/objaverse_partial",folder_name)

        save_dataset(instance_savedir, folder_name, images_tensor, focals, poses, pts3d)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Run inference in parallel across multiple GPUs
def run_inference(rank, world_size, urls):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # Set the device for each process

    # Initialize model and move it to the current device
    model = DusterRun().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Set up WebDataset with nodesplitter
    batch_size = 4

    rank_urls = urls[rank::world_size]

    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .map(postprocess_fn)
        .batched(batch_size)
    )

    # Configure DataLoader with DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # outputs = []
    with torch.no_grad():
        for batch in data_loader:
            import pdb; pdb.set_trace()
            batch = batch.to(rank)
            model(batch)
            # outputs.append(output.cpu())  # Gather results on CPU

    cleanup()
    # return torch.cat(outputs)


if __name__ == "__main__":

    # genwarp_cfg = dict(
    #     pretrained_model_path='./checkpoints',
    #     checkpoint_name='multi1',
    #     half_precision_weights=False
    # )

    # genwarp_nvs = GenWarp(cfg=genwarp_cfg)

    ENDPOINT_URL = 'https://storage.clova.ai'

    os.environ['AWS_ACCESS_KEY_ID'] = "AUIVA2ODFS9S2YDD0A75"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "VDIVVqIC9FCC0GmOQ2nNy3o7NjkWVqC4oTDOz3mM"
    os.environ['S3_ENDPOINT_URL'] = ENDPOINT_URL

    urls = [f's3://generation-research/objaverse_z/objaverse_rendering_train_{num:06}.tar' for num in range(100)]

    # add awscli command to urls
    urls = [f'pipe:aws --endpoint-url={ENDPOINT_URL} s3 cp {url} -' for url in urls]

    batch_size = 8 

    world_size = torch.cuda.device_count()  # Number of GPUs available

    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    mp.spawn(run_inference,
            args=(world_size, urls),
            nprocs=world_size,
            join=True)

    p = 0

    for i, batch in enumerate(dataset):

        try:

            images_tensor = torch.cat((batch["support_rgbs"], batch["query_rgbs"]), dim=0)
            # images has dimension (12, 3, 512, 512)

            images = [dict(img = image[None,...], idx=i, instance=i, true_shape=np.array([[512,512]])) for i, image in enumerate(images_tensor) ]

            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=batch_size)
            
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

            folder_name = batch["instance_name"]

            instance_savedir = os.path.join("/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/multiview-gen/genwarp/objaverse_partial",folder_name)

            save_dataset(instance_savedir, folder_name, images_tensor, focals, poses, pts3d)

        except:
            pass
        # import pdb; pdb.set_trace()



        # # combined_results = reprojector(pts, pts_feat, proj_pose, proj_pose_inv, proj_fov, device=device, coord_channel=3 + coord_channel)
        
        # proj_results = combined_results[...,:coord_channel]
        # warped_results = combined_results[...,coord_channel:]
        
        


        # import pdb; pdb.set_trace()
        # print(f"Batch {i + 1}:")




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






