import os
import glob
import torch
import numpy as np
import imageio

from PIL import Image
from time import gmtime, strftime
from tqdm import tqdm

import webdataset as wds
import random
import json

from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from genwarp import GenWarp
from genwarp.ops import camera_lookat, get_projection_matrix
import torchvision.transforms as transforms

import torch.nn.functional as F
import argparse

from genwarp.dust3r.inference import inference
from genwarp.dust3r.model import AsymmetricCroCo3DStereo
# from genwarp.dust3r.utils.image import load_images
from genwarp.dust3r.image_pairs import make_pairs
from genwarp.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import multiprocessing

# from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

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


def save_dataset(folder_name, instance, images, focals, poses, pts3d, confidence):
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

    # import pdb; pdb.set_trace()

    try:
        # Save images and point cloud maps
        for i in range(len(images)):
            # Convert image tensor to a PIL image (assuming images are in range [0, 1] or [0, 255])
            img = images[i]
            # .permute(1, 2, 0).cpu().numpy()  # Convert to (512, 512, 3) format
            # img = (img * 255).astype(np.uint8)  # Convert to uint8 for saving as PNG
            # img_pil = Image.fromarray(img)

            # Save the image as PNG
            img_path = os.path.join(folder_name, f'image_{i}.png')
            save_image(img, img_path)

            # Save the point cloud map as .npy (preserving original values)
            point_cloud = pts3d[i].detach().cpu().numpy()  # Convert to numpy array (512, 512, 3)

            # Add the focal length and pose to annotation data
            focal = focals[i].detach().cpu().numpy()  # Convert focal length to numpy
            pose = poses[i].detach().cpu().numpy()  # Convert pose to numpy
            conf = confidence[i].detach().cpu().numpy()  # Convert confidence value to numpy

            # Store the data in a dictionary for later saving
            annotation_data.append({
                'img': point_cloud,
                'focal': focal,
                'pose': pose,
                'conf': conf
            })

        # Save the annotation data as a numpy file
        annotation_file_path = os.path.join(folder_name, 'annotations.npy')
        np.save(annotation_file_path, annotation_data)

    except:
        pass

    # import pdb; pdb.set_trace()

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


def postprocess_fn(sample):

    # Read the annotation, retrieve views

    try:
        category, instance = sample["__key__"].split("+")
        # anno_key = "annotations_" + instance 

        image_dict = json.loads(sample["annotations"])
        frame_keys = [key for key in image_dict.keys()]
        # frame_keys_list = [f.split(" ")[0] + ".png" for f in frame_data]

        # random.shuffle(frame_keys)

        img_data = []

        num_splits = 3
        split_len = len(frame_keys) // num_splits

        per_batch = 6

        frame_lists = [frame_keys[i*split_len:(i+1)*split_len] for i in range(num_splits)]

        process_lists = []

        for le in frame_lists:
            random.shuffle(le)
            le = le[:per_batch]
            le = sorted(list(map(lambda x:int(x), le)))
            process_lists.append(le)

        for i, pro in enumerate(process_lists):
            if i <= len(process_lists)-2:
                process_lists[i+1][0] = pro[-1]

        for frame_data in process_lists:
            for single_frame_data in frame_data:

                image_key = "img_" + f"{int(single_frame_data):03}" + ".png"

                img = sample[image_key]

                img_data.append(img)

        # import pdb; pdb.set_trace()

        output = dict(batch0=img_data[:per_batch], batch1=img_data[per_batch:2*per_batch], batch2=img_data[2*per_batch:]) 
    
    except:
        category = None
        instance = None
        output = None
    
    return category, instance, output



def handle_error(exn):
    print(f"Skipping problematic file due to error: {exn}")
    return None

def main(args):
    # Initialize GenWarp configuration
    ENDPOINT_URL = 'https://storage.clova.ai'

    visualize = args.visualize

    if visualize:
        genwarp_cfg = dict(
            pretrained_model_path='./checkpoints',
            checkpoint_name='multi_best',
            half_precision_weights=False,
            embedder_input_dim=3
        )

        genwarp_nvs = GenWarp(cfg=genwarp_cfg)

    os.environ['AWS_ACCESS_KEY_ID'] = "AUIVA2ODFS9S2YDD0A75"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "VDIVVqIC9FCC0GmOQ2nNy3o7NjkWVqC4oTDOz3mM"
    os.environ['S3_ENDPOINT_URL'] = ENDPOINT_URL

    # Load images
    N = args.gpu_num
    num_list = torch.arange(N * 100, (N+1) * 100)

    urls = [f's3://generation-research/co3d_light/co3d_{num:04}.tar' for num in num_list]

    # add awscli command to urls
    urls = [f'pipe:aws --endpoint-url={ENDPOINT_URL} s3 cp {url} -' for url in urls]

    dataset = (
            wds.WebDataset(urls, shardshuffle=False, handler=wds.ignore_and_continue)
            .decode("pil")
            .map(postprocess_fn)
    )

    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(args.device)

    for data in dataset:

        category, instance, image_batches = data
        split = "train" if random.random() < 0.93 else "val"

        try:
            batches = image_batches.keys()
        except:
            continue

        try:
            for k, batch in enumerate(batches):
                src_images = []
                image_tensors = []
                img_dict = image_batches[batch]

                for i, image in enumerate(img_dict):
                    x = to_tensor(image)[None,...]
                    img_dict = dict(img = x, true_shape=np.array([[512,512]]), idx=i, instance=str(i))
                    src_images.append(img_dict)
                    image_tensors.append(x)

                pairs = make_pairs(src_images, scene_graph='complete', prefilter=None, symmetrize=True)
                output = inference(pairs, model, args.device, batch_size=args.batch_size)

                # Extract predictions and setup scene alignment
                # view1, pred1 = output['view1'], output['pred1']
                # view2, pred2 = output['view2'], output['pred2']
            
                scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
                loss = scene.compute_global_alignment(init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr)

                # Gather focal, pose, and 3D point information
                focals = [k.to(args.device) for k in scene.get_focals()]
                poses = [k.to(args.device) for k in scene.get_im_poses()]
                pts3d = [k.to(args.device) for k in scene.get_pts3d()]
                confidence = [k.to(args.device) for k in scene.im_conf]

                if visualize:
                    device = pts3d[0].device

                    poses = torch.stack(poses)
                    focals = torch.stack(focals)

                    src_idx = [0,1,2,3]
                    target_idx = 4
                    new_target_pose = poses[target_idx]

                    images = dict(ref=[image_tensors[k].to(device) for k in src_idx], tgt=image_tensors[target_idx].to(device))
                    correspondence = dict(ref=torch.cat([pts3d[k][None,...].to(device) for k in src_idx]), tgt=pts3d[target_idx])

                    ref_camera=dict(pose=poses[src_idx].to(device), 
                                    focals=focals[src_idx].to(device), 
                                    orig_img_size=torch.tensor([512, 512]).to(device))
                
                    tgt_camera=dict(pose=new_target_pose[None,...].to(device),
                                    focals=focals[target_idx][None,...].to(device),
                                    orig_img_size=torch.tensor([512, 512]).to(device))
                    
                    camera_info = dict(ref=ref_camera, tgt=tgt_camera)

                    renders = genwarp_nvs(
                        images = images,
                        correspondence = correspondence,
                        camera_info = camera_info,
                        dataset = "realestate"
                    )

                    # import pdb; pdb.set_trace()

                instance_savedir = os.path.join("/mnt/tmp/co3d_duster_conf", split, category, instance, "viewbatch_" + str(k))

                save_dataset(instance_savedir, instance, image_tensors, focals, poses, pts3d, confidence)

                # import pdb; pdb.set_trace()
        
        except:
            continue
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GenWarp and DUSt3R model inference.")

    # Configuration arguments
    parser.add_argument("--pretrained_model_path", type=str, default='./checkpoints', help="Path to pretrained model.")
    parser.add_argument("--checkpoint_name", type=str, default='multi1', help="Checkpoint name.")
    parser.add_argument("--half_precision_weights", type=bool, default=False, help="Use half-precision weights.")

    # Model and dataset arguments
    parser.add_argument("--gpu_num", type=int, default=0, help="GPU number.")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualizing option.")
    parser.add_argument("--num_processes", type=int, default=5, help="Number of processes for a single GPU.")
    parser.add_argument("--model_name", type=str, default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", help="Model name or path.")
    # parser.add_argument("--image_files", nargs='+', required=True, help="List of image file paths.")

    # Inference settings
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for inference.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--schedule", type=str, default='cosine', help="Learning rate schedule.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--niter", type=int, default=300, help="Number of iterations.")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames for the video.")

    # Output folder arguments
    parser.add_argument("--syn_folder", type=str, default="syn_mask_yes_norm", help="Folder name for synthesized frames.")
    parser.add_argument("--warp_folder", type=str, default="warp", help="Folder name for warped frames.")
    parser.add_argument("--corres_folder", type=str, default="corres", help="Folder name for correspondence frames.")

    args = parser.parse_args()
    main(args)






