import os
import sys
import glob
import torch
import numpy as np
import imageio
import random
import json
import sys
import pdb

from PIL import Image
from time import gmtime, strftime
from tqdm import tqdm

from torch.utils.data import Dataset

import open3d as o3d

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

class Co3DDataset(Dataset):
    def __init__(self, co3d_path, num_random_samples=3, transform=None):
        # Load the dataset annotations from the json file
        
        self.co3d_path = co3d_path  # Root directory where the data is stored
        self.transform = transform  # Optional transforms (for images, etc.)
        self.new_category = True
        self.num_random_samples = num_random_samples
        self.num_viewpoints = num_random_samples

        target_categories = ["car", "motorcycle"]

        self.instances = []
        self.eval_instances = []

        # if new_annotation
        for category in target_categories:

            folders = glob.glob(os.path.join(self.co3d_path, category, "*/"))[2:]
            instances = [category + "/" + os.path.basename(os.path.normpath(folder)) for folder in folders]

            partition = int(0.9 * len(instances))

            self.instances += instances[:partition]
            self.eval_instances += instances[partition:]


    def __len__(self):
        # The length of the dataset is the number of frames in the JSON
        return len(self.instances)

    def __getitem__(self, idx):
        # Each frame in the dataset is indexed by '1', '2', '3', etc.

        # Random category
        category_instance_name = self.instances[idx]

        # category_instance_name = "motorcycle/216_22798_47409"

        annotation_file = os.path.join(self.co3d_path, category_instance_name, 'frame_annotations.json')

        with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)

        keys = [*self.annotations.keys()]

        # ForkedPdb().set_trace()

        frame_idx_list = random.sample(range(len(self.annotations)), k=self.num_viewpoints)

        l_img = []
        l_depth = []
        l_mask = []
        l_R = []
        l_T = []
        l_K = []
        l_focal = []
        l_prin = []

        # frame_idx_list = [5, 15, 25]

        for frame_idx in frame_idx_list:

            frame_data = self.annotations[keys[frame_idx]]

            # Load the image, depth, and mask
            img_path = os.path.join(self.co3d_path, frame_data['image']['path'])
            depth_path = os.path.join(self.co3d_path, frame_data['depth']['path'])
            mask_path = os.path.join(self.co3d_path, frame_data['depth']['mask_path'])
            # mask_path = os.path.join(self.co3d_path, frame_data['mask']['path'])

            # Extract viewpoint data (e.g., rotation matrix, translation vector)
            viewpoint = frame_data['viewpoint']
            R = torch.tensor(viewpoint['R'], dtype=torch.float32)
            T = torch.tensor(viewpoint['T'], dtype=torch.float32).reshape(3,1)
            focal_length = torch.tensor(viewpoint['focal_length'], dtype=torch.float32)
            principal_point = torch.tensor(viewpoint['principal_point'], dtype=torch.float32)

            # ForkedPdb().set_trace()

            K = torch.tensor([[focal_length[0], 0., principal_point[0]],
                              [0., focal_length[1], principal_point[1]],
                              [0.,               0.,                1.]]).to(R.device)

            image = Image.open(img_path).convert("RGB")
            depth = Image.open(depth_path)
            depth = _load_16big_png_depth(depth_path)
            mask = Image.open(mask_path)

            img_shape = torch.tensor([image.size[1],image.size[0]]).to(R.device)

            size = torch.min(img_shape)
            sq_factor = 512 / size 

            # denorm_factor = torch.tensor([
            #     [sq_factor * size / 2., 0., sq_factor * img_shape[1] / 2.],
            #     [0., sq_factor * size / 2., sq_factor * img_shape[0] / 2.],
            #     [0., 0., 1.]
            # ]).to(R.device)

            denorm_factor = torch.tensor([
                [sq_factor * size / 2., 0., 512 / 2.],
                [0., sq_factor * size / 2., 512 / 2.],
                [0., 0., 1.]
            ]).to(R.device)

            

            # denorm_factor = torch.tensor([
            #     [img_shape[0] / 2., 0., img_shape[0] / 2.],
            #     [0., img_shape[1] / 2., img_shape[1] / 2.],
            #     [0., 0., 1.]
            # ]).to(R.device)

            # denorm_factor = torch.tensor([
            #     [shape / 2., 0., shape / 2.],
            #     [0., shape / 2., shape / 2.],
            #     [0., 0., 1.]
            # ]).to(R.device)

            K = denorm_factor @ K

            # Apply optional transformations (e.g., resizing, normalization, etc.)
            if self.transform:
                image = self.transform(image)
                depth = self.transform(depth)
                mask = self.transform(mask)

            l_img.append(image)
            l_depth.append(depth)
            l_mask.append(mask)
            l_R.append(R)
            l_T.append(T)
            l_K.append(K)
            l_focal.append(focal_length)
            l_prin.append(principal_point)

        image = torch.stack(l_img)
        depth = torch.stack(l_depth)
        mask = torch.stack(l_mask)
        R = torch.stack(l_R)
        T = torch.stack(l_T)
        K = torch.stack(l_K)
        focal_length = torch.stack(l_focal)
        orig_img_size = torch.stack([img_shape] * self.num_viewpoints)
        # principal_point = torch.stack(l_prin)

        # point_cloud = o3d.io.read_point_cloud(os.path.join(self.co3d_path, category_instance_name,"pointcloud.ply"))
        # points_np = np.asarray(point_cloud.points)
        # points_rgb_np = np.asarray(point_cloud.colors)
        # points_tensor = torch.tensor(points_np, dtype=torch.float32)
        # points_rgb = torch.tensor(points_rgb_np, dtype=torch.float32)
        
        # sample_indices = torch.randperm(points_tensor.size(0))[:50000]
        # sampled_points = points_tensor[sample_indices]
        # sampled_rgb = points_rgb[sample_indices]

        # ForkedPdb().set_trace()

        # Return a dictionary of the loaded data
        sample = {
            'image': image,
            'depth': depth,
            'mask': mask,
            'R': R,
            'T': T,
            'K': K,
            'focal_length': focal_length,
            'principal_point': principal_point,
            'instance_name': [category_instance_name],
            'orig_img_size' : orig_img_size,
        }

        return sample