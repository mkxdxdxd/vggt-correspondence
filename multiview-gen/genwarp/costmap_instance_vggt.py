import os
import torch
from functools import partial
# import webdataset as wds
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from genwarp.utils import (
#     reprojector,
#     ndc_rasterizer,
#     get_rays,
#     pose_to_ray,
#     one_to_one_rasterizer,
#     mesh_rendering,
#     features_to_world_space_mesh,
#     torch_to_o3d_mesh,
#     torch_to_o3d_cuda_mesh,
#     compute_plucker_embed,
#     postprocess_co3d,
#     postprocess_realestate,
#     postprocess_combined,
#     postprocess_vggt,
#     PointmapNormalizer,
#     UncertaintyLoss
# )
# from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path
from typing import Union, Tuple, Optional
import json
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
# import torchvision
# import wandb

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
from vggt.models.vggt_feature import VGGT

# from croco.models.croco import CroCoNet
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.colors as mpl_colors
from croco.models.croco import CroCoNet
from glob import glob

# dataset_length = 1000
# epoch = 10000
# shardshuffle=False
# resampled=False

# uniform_sampling = True
# sampling_views = 12
# num_ref_viewpoints = 11
# view_range = 30 # sample 3 views from 30 view range
# train_bs = 1
# world_size = 1
model_type = vggt"
if model_type == "croco": 
    checkpoint_path = '/workspace/minkyung/multiview-gen/genwarp/croco/checkpoint/CroCo_V2_ViTLarge_BaseDecoder.pth'
target_position = [11, 26] # of 37, 37
save_path_origin = '/workspace/minkyung/multiview-gen/costmap_feature_layer'



# def get_attn_map(attn_layer, background, height, width):
#     resize = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
#     # 1t1: target_image_clone / 1t2: source_iamge_clone
#     h = int((height+0.5) / 16. * 16)
#     w = int((width+0.5) / 16. * 16)
#     attn_layer = attn_layer.reshape(16, 16, 16, 16)[h, w]
#     # max_idx = np.unravel_index(np.argmax(attn_layer, axis=None), attn_layer.shape)
#     # attn_layer[max_idx[0]][max_idx[1]] = 0
#     attn_layer = resize(attn_layer.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()
#     normalizer = mpl.colors.Normalize(vmin=attn_layer.min(), vmax=attn_layer.max())
#     mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
#     colormapped_im = (mapper.to_rgba(attn_layer[:,:,0])[:, :, :3] * 255).astype(np.uint8)
#     attn_map = cv2.addWeighted(background.copy(), 0.3, colormapped_im, 0.7, 0)
    
#     # Find max attention location
#     max_idx = np.unravel_index(np.argmax(attn_layer, axis=None), attn_layer.shape)

#     # Scale max position to resized image dimensions
#     scale_x = 512 / attn_layer.shape[1]  # 512 / 14
#     scale_y = 512 / attn_layer.shape[0]  # 512 / 14
#     max_x = int(max_idx[1] * scale_x)  # Scale width index
#     max_y = int(max_idx[0] * scale_y)  # Scale height index

#     # Draw dot on max attention point
#     attn_map = cv2.circle(attn_map, (max_x, max_y), 10, (255, 255, 255), -1)  # Blue dot
#     attn_map = cv2.circle(attn_map, (max_x, max_y), 6, (255, 0, 0), -1)
#     return attn_map



def get_cosine_similarity(feature1, feature2, ref1, height, width):
    """
    Compute cosine similarity between a selected feature in feature1 and all features in feature2,
    and overlay the resulting 'viridis'-style heatmap on the reference image (ref1).
    
    Args:
        feature1: torch.Tensor [256, 1024]
        feature2: torch.Tensor [256, 1024]
        ref1: numpy.ndarray [H, W, 3], RGB format, values in [0, 1] or [0, 255]
        height, width: int in range 0–15 selecting a feature at 16x16 grid

    Returns:
        PIL.Image with viridis heatmap overlaid on ref1
    """
    # 1. Select the query feature
    if model_type == "croco":
        height = int((height+0.5) / 37 * 14) 
        width = int((width+0.5) / 37 * 14) 
    size = int(feature1.shape[0]**0.5)
    f1_reshaped = feature1.view(size, size, -1)
    query_feat = f1_reshaped[height, width].unsqueeze(0)  # shape [1, 1024]
    query_feat = query_feat / torch.linalg.norm(query_feat, axis = 1, keepdims=True)
    feature2 = feature2 / torch.linalg.norm(feature2, axis = 1, keepdims=True)
    # sim = torch.mm(feature2 ,query_feat.transpose(0,1))
    # sim = sim/10
    # 2. Compute negative squared Euclidean distance
    diff = feature2 - query_feat  # [256, 1024]
    dist_sq = torch.sum(diff ** 2, dim=1)  # [256]

    # Optional: apply Gaussian kernel (RBF-like similarity)
    sigma = 0.1
    sim = torch.exp(-dist_sq / (2 * sigma ** 2))  # [256]

    # 3. Reshape to similarity map
    sim_map = sim.view(size, size).cpu().numpy()
    
    # 4. Resize similarity map to match image size
    H, W, _ = ref1.shape
    sim_map_resized = cv2.resize(sim_map, (W, H), interpolation=cv2.INTER_LINEAR)

    max_idx = np.unravel_index(np.argmax(sim_map_resized), sim_map_resized.shape)
    height_circle, width_circle = max_idx  # note: (row, col)

    # 2. Apply matplotlib colormap ('viridis')
    normalizer = mpl_colors.Normalize(vmin=sim_map_resized.min(), vmax=sim_map_resized.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    colormapped_im = (mapper.to_rgba(sim_map_resized)[:, :, :3] * 255).astype(np.uint8)

    # 3. Prepare background image
    if ref1.max() <= 1.0:
        bg = (ref1 * 255).astype(np.uint8)
    else:
        bg = ref1.astype(np.uint8)

    # 4. Blend colormap and background
    blended = cv2.addWeighted(bg.copy(), 0.3, colormapped_im, 0.7, 0)

    # 5. Draw outer white circle and inner red circle at the max point
    tgt_circle = cv2.circle(blended.copy(),
                            (width_circle, height_circle),  # (x, y)
                            radius=10,
                            color=(255, 255, 255),
                            thickness=-1)
    tgt_circle = cv2.circle(tgt_circle,
                            (width_circle, height_circle),
                            radius=6,
                            color=(255, 0, 0),
                            thickness=-1)

    # 6. Return the final PIL Image
    return Image.fromarray(tgt_circle)

# def get_cosine_similarity(feature1, feature2, background, height, width):
#     resize = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
#     # 1t1: target_image_clone / 1t2: source_iamge_clone
#     h = int((height+0.5) / 16. * 16)
#     w = int((width+0.5) / 16. * 16)
#     attn_layer = attn_layer.reshape(16, 16, 16, 16)[h, w]
#     # max_idx = np.unravel_index(np.argmax(attn_layer, axis=None), attn_layer.shape)
#     # attn_layer[max_idx[0]][max_idx[1]] = 0
#     attn_layer = resize(attn_layer.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()
#     normalizer = mpl.colors.Normalize(vmin=attn_layer.min(), vmax=attn_layer.max())
#     mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
#     colormapped_im = (mapper.to_rgba(attn_layer[:,:,0])[:, :, :3] * 255).astype(np.uint8)
#     attn_map = cv2.addWeighted(background.copy(), 0.3, colormapped_im, 0.7, 0)
    
#     # Find max attention location
#     max_idx = np.unravel_index(np.argmax(attn_layer, axis=None), attn_layer.shape)

#     # Scale max position to resized image dimensions
#     scale_x = 512 / attn_layer.shape[1]  # 512 / 14
#     scale_y = 512 / attn_layer.shape[0]  # 512 / 14
#     max_x = int(max_idx[1] * scale_x)  # Scale width index
#     max_y = int(max_idx[0] * scale_y)  # Scale height index

#     # Draw dot on max attention point
#     attn_map = cv2.circle(attn_map, (max_x, max_y), 10, (255, 255, 255), -1)  # Blue dot
#     attn_map = cv2.circle(attn_map, (max_x, max_y), 6, (255, 0, 0), -1)
#     return attn_map


# train_dataloader = DataLoader(train_dataset, num_workers=world_size, batch_size=train_bs, persistent_workers=True)
if model_type == 'vggt':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        dtype = torch.float16
        dift = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
elif model_type =='dino':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    dift = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
elif model_type == 'croco':
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, 'cpu')
    dift = CroCoNet( **ckpt.get('croco_kwargs',{})).to(device)
    dift.eval()
    msg = dift.load_state_dict(ckpt['model'], strict=True)

def load_and_crop(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size

    # Resize to 512x512 and convert to tensor
    if model_type == "croco":
        transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
    else: 
        transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((518, 518))
        ])
    transform_orig = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform_orig(img), transform_input(img)
    
# Load and crop both images
path1 = '/workspace/minkyung/dift/SPair-71k/JPEGImages/dog/2010_000608.jpg'
img1_orig, img1 = load_and_crop(path1)

# Image directory
img_dir = '/workspace/minkyung/dift/SPair-71k/JPEGImages/cat/'
image_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))  # Get all .jpg files


for layer in [4, 11, 17, 23]:
    print(f"processing layer {layer}")
    # Apply to all images
    for i, path in tqdm(enumerate(image_paths)):
        img2_orig, img2 = load_and_crop(path)

        # Convert tensor to PIL image
        # print("loading finished!, now feature extraction!")
        category = os.path.basename(os.path.dirname(path))
        save_path = f'{save_path_origin}/{category}_{target_position[0]}_{target_position[1]}/layer_{layer}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if i == 0:
            #target marking
            _, h, w = img1_orig.shape
            height, width = target_position[0], target_position[1]
            height_circle = int((height+0.5) / 37 * h) 
            width_circle = int((width+0.5) / 37. * w) 
            tgt = (img1_orig.squeeze(0).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8) # [H, W, 3]

            tgt_circle = cv2.circle(tgt.copy(), (width_circle, height_circle), 10, (255, 255, 255), -1)        
            tgt_circle = cv2.circle(tgt_circle, (width_circle, height_circle), 6, (0, 0, 255), -1)    
            tgt_circle = Image.fromarray((tgt_circle).astype(np.uint8))
            filename = os.path.splitext(os.path.basename(path1))[0]
            tgt_circle.save(f'{save_path}/tgt_{filename}.png')

        ref1 = (img2_orig.squeeze(0).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
        with torch.inference_mode(): 
            if model_type == "vggt":
                feature1 = dift(img1.unsqueeze(0).to(device), layer=layer)
                feature2 = dift(img2.unsqueeze(0).to(device), layer=layer)
                feature1 = feature1.squeeze(0).permute(1,2,0).view(-1, 1024)
                feature2 = feature2.squeeze(0).permute(1,2,0).view(-1, 1024)
            elif model_type == 'croco':
                feature1, _, _ = dift._encode_image(img1.unsqueeze(0).to(device)) # [1, 196, 1024]
                feature2, _, _ = dift._encode_image(img2.unsqueeze(0).to(device))
                feature1 = feature1.squeeze(0)
                feature2 = feature2.squeeze(0)
            elif model_type == 'dino':
                feature1 = dift.forward_features(img1.unsqueeze(0).to(device))['x_norm_patchtokens'] # [1, 1369, 768]
                feature2 = dift.forward_features(img2.unsqueeze(0).to(device))['x_norm_patchtokens'] # [1, 1369, 768]
                feature1 = feature1.squeeze(0)
                feature2 = feature2.squeeze(0)
        feature_map = get_cosine_similarity(feature1, feature2, ref1, height, width)
        feature_map.save(f'{save_path}/attn_layer{layer}_{Path(path).stem}.png')

# idx = [4, 11, 17, 23]  # Layer indices

# for i, frame in enumerate(attn_frames):
#     # If it's a PIL image, just save it directly
#     if isinstance(frame, Image.Image):
#         img = frame
#     else:
#         # If it's a torch tensor, convert to numpy
#         if isinstance(frame, torch.Tensor):
#             frame = frame.detach().cpu().numpy()

#         # Normalize if values are in [0, 1]
#         if frame.max() <= 1.0:
#             frame = (frame * 255).astype(np.uint8)
#         else:
#             frame = frame.astype(np.uint8)

#         img = Image.fromarray(frame)

#     img.save(f'{save_path}/attn_layer{idx[i]}_ref1_dog.png')
    

#save_attn_frames(attn_frames, f'{save_path}/attn_layers_ref1_dog.png') # (4, 512, 512, 3)

plt.close('all')

    # attn_mean = attn_mean.mean(dim=0) #[256, 256]
    # attn_map_mean = get_cosine_similarity(attn_mean, ref1, height, width)
    # attn_map1 = Image.fromarray(attn_map_mean)
    # attn_map1.save(f'{save_path}/attn_ref1.png')
    # #ref1 processing  
    # attn = torch.stack(attn_map2, dim=0).squeeze(1).cpu() #[4, 16, 256, 256], [depth, head, heigh, width]
    # attn_mean = attn.mean(dim=1)  #[4, 256, 256]
    # attn_frames = []
    # for j in range(len(attn_map2)): 
    #     attn_map = get_attn_map(attn_mean[j], ref1, height, width)
    #     attn_frames.append(attn_map)
    # save_attn_frames(attn_frames, f'{save_path}/attn_layers_ref1.png') # (12, 512, 512, 3)

    # attn_mean = attn_mean.mean(dim=0) #[256, 256]
    # attn_map_mean = get_attn_map(attn_mean, ref1, height, width)
    # attn_map1 = Image.fromarray(attn_map_mean)
    # attn_map1.save(f'{save_path}/attn_ref1.png')
    
    # #ref2 processing  
    # attn = torch.stack(attn_map3, dim=0).squeeze(1).cpu() #[4, 16, 256, 256], [depth, head, heigh, width]
    # attn_mean = attn.mean(dim=1)  
    # attn_frames = []
    # for j in range(len(attn_map3)): 
    #     attn_map = get_attn_map(attn_mean[j], ref2, height, width)
    #     attn_frames.append(attn_map)
    # save_attn_frames(attn_frames, f'{save_path}/attn_layers_ref2.png') # (12, 512, 512, 3)
    
    # attn_mean = attn_mean.mean(dim=0) 
    # attn_map_mean = get_attn_map(attn_mean, ref2, height, width)
    
    # attn_map2 = Image.fromarray(attn_map_mean)
    # attn_map2.save(f'{save_path}/attn_ref2.png')
    
    # # Assume tgt_circle, attn_map1, attn_map2 are all PIL Images
    # images = [tgt_circle, attn_map1, attn_map2]

    # # Get total width and max height
    # total_width = sum(img.width for img in images)
    # max_height = max(img.height for img in images)

    # # Create a new blank image
    # combined = Image.new('RGB', (total_width, max_height))

    # # Paste images next to each other
    # x_offset = 0
    # for img in images:
    #     combined.paste(img, (x_offset, 0))
    #     x_offset += img.width

    # # Save the combined image
    # combined.save(f'{save_path}/combined.png')
    # # ref = Image.fromarray((ref).astype(np.uint8))
    # # ref.save(f'{save_path}/ref.png')
    # # tgt = Image.fromarray((tgt).astype(np.uint8))
    # # tgt.save(f'{save_path}/tgt.png')
    
    # plt.close('all') 



