import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pdb
import sys
import torch

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


class RealEstateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the instances.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.instances = sorted(os.listdir(root_dir))  # List all instance directories

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # Get the directory for the specific instance
        instance_dir = os.path.join(self.root_dir, self.instances[idx])
        
        # Load all images in the instance directory
        images = []

        for file_name in sorted(os.listdir(instance_dir)):
            if file_name.endswith(".png"):  # Adjust the extension if needed
                image_path = os.path.join(instance_dir, file_name)
                image = Image.open(image_path).convert("RGB")  # Load and convert to RGB
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        
        # Load the .npy file (dictionary) in the instance directory
        # npy_files = [f for f in os.listdir(instance_dir) if f.endswith(".npy")]
        # if npy_files:
        npy_file_path = os.path.join(instance_dir, "annotations.npy")  # Assuming one .npy file per instance
        npy_data = np.load(npy_file_path, allow_pickle=True) # Load as dictionary
        # else:
        #     npy_data = None

        pts = []
        focals = []
        pose = []

        # ForkedPdb().set_trace()

        for i in range(len(images)):
            pts.append(torch.from_numpy(npy_data[i]['img']))
            focals.append(torch.from_numpy(npy_data[i]['focal']))
            pose.append(torch.from_numpy(npy_data[i]['pose']))

        
        image_batch = torch.stack(images)
        points = torch.stack(pts)
        focals = torch.stack(focals)
        pose = torch.stack(pose)

        # points = npy_data["img"]
        # focals = npy_data["focal"]
        # pose = npy_data["pose"]

        # Return all images and the dictionary from the .npy file
        return {"image": image_batch, "points": points, "focals": focals, "pose": pose}