import os
import glob
import torch
import numpy as np
import imageio

from PIL import Image
from time import gmtime, strftime
from tqdm import tqdm

from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from genwarp import GenWarp
from genwarp.ops import camera_lookat, get_projection_matrix
import torchvision.transforms as transforms

import torch.nn.functional as F

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from genwarp.utils import (
    reprojector,
    ndc_rasterizer,
    one_to_one_rasterizer
)

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

    image_files = [
                #   "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/46012633.png",
                #   "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/49449400.png",
                  "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/46513133.png",
                  "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/46946900.png",
                  "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/49215833.png",
                # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/RealEstate10K_Downloader/dataset_old/train/0a9e82926ed00bec/49482767.png",
                ]

    # Run Dust3r
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    transform = transforms.Compose([
        transforms.Resize((512, 512))  # Resize to (512, 512)
    ])

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(image_files, size=512)

    import pdb; pdb.set_trace()

    src_images = [transform(to_tensor(Image.open(image_file).convert('RGB'))[None].cuda()) for image_file in image_files]

    for i, img in enumerate(images):
        img["img"] = transform(img['img'])
        img["true_shape"] = np.array([[512,512]])
    
    # import pdb; pdb.set_trace()

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

    focals = torch.stack([k.to(device) for k in focals])
    poses = torch.stack([k.to(device) for k in poses])
    pts3d = [k.to(device) for k in pts3d]

    # For camera transition video

    eye_pos_start = torch.tensor([[0., 0., 0.]]).to(device)
    lookat_start = torch.tensor([[-1., 0., 0.]]).to(device)

    eye_pos_end = torch.tensor([[0.02, -0.02, 0.02]]).to(device)
    lookat_end = torch.tensor([[-0.7, -0.2, 0.]]).to(device)

    num_frames = 20

    alphas = torch.linspace(0, 1, steps=num_frames).to(device)

    frames = []
    warped_frames = []
    corresponding_frames = []

    for alpha in tqdm(alphas):

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
        # tgt_idx = [target_idx]
        tgt_idx = target_idx
        # Preparing input information for GenWarp

        images = dict(ref=[src_images[k] for k in src_idx], tgt=src_images[target_idx])
        correspondence = dict(ref=torch.cat([pts3d[k][None,...] for k in src_idx]), tgt=pts3d[target_idx])

        points = correspondence["ref"].reshape(-1,3)

        new_pts = (points - points.min(dim=0, keepdim=True)[0]) / (points.max(dim=0, keepdim=True)[0] - points.min(dim=0, keepdim=True)[0])

        new_pts = new_pts.reshape(2,512,512,3)

        ref_camera=dict(pose=poses[src_idx], 
                        focals=focals[src_idx], 
                        orig_img_size=torch.tensor([512, 512]).to(device))
    
        tgt_camera=dict(pose=new_target_pose,
                        focals=focals[target_idx],
                        orig_img_size=torch.tensor([512, 512]).to(device))
        # import pdb; pdb.set_trace()

        # tgt_camera=dict(pose=poses[tgt_idx],
        #                 focals=focals[tgt_idx],
        #                 orig_img_size=torch.tensor([512, 512]).to(device))
        
        camera_info = dict(ref=ref_camera, tgt=tgt_camera)

        src_corr = correspondence["ref"]
        pts_locs = src_corr.reshape(-1,3)

        # import pdb; pdb.set_trace()

        combined_results = reprojector(pts_locs, torch.cat((new_pts.reshape(-1,3), torch.cat(images["ref"]).permute(0,2,3,1).reshape(-1,3)), dim=-1), tgt_camera, device=device, coord_channel=6)

        # save_image(torch.stack(src_images), "images.png")
        # save_image(new_pts[0].permute(2,0,1), "t_source_1.png")
        # save_image(new_pts[1].permute(2,0,1), "t_source_2.png")

        # import pdb; pdb.set_trace()
        hey = combined_results[0][...,:3]
        ho = combined_results[0][...,3:]

        # save_image(hey[0].permute(2,0,1) , "target.png")
        # save_image(ho[0].permute(2,0,1) , "target.png")

        # import pdb; pdb.set_trace()

        # renders = genwarp_nvs(
        #     images = images,
        #     correspondence = correspondence,
        #     camera_info = camera_info
        # )


        # # Outputs.
        # renders['synthesized']     # Generated image.
        # renders['warped']

        # frames.append(renders['synthesized'][0])
        warped_frames.append(hey.permute(2,0,1).cpu().detach())
        corresponding_frames.append(ho.permute(2,0,1).cpu().detach())

    now = strftime("%m_%d_%H_%M_%S", gmtime())
    # make_video(frames, now, folder_name="syn_mask_yes_norm")
    make_video(warped_frames, now, folder_name="warp")
    make_video(corresponding_frames, now, folder_name="corres")






