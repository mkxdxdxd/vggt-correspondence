import os
from typing import Dict, Union, Tuple
from functools import partial

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

import random

def create_dataset(urls, length, batch_size, resampled, stage, postprocess_fn, world_size, force_no_shuffle=False):
    
    shardshuffle = 100 if stage == "train" and not force_no_shuffle else None

    dloader_length = length // (batch_size * world_size)
    dataset_length = dloader_length * batch_size

    dataset = (
        wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,
            shardshuffle=shardshuffle,
            detshuffle=True, # only for training
            resampled=resampled,
            handler=wds.ignore_and_continue,
        )
        .shuffle(
            size=(1 if shardshuffle is None else shardshuffle * 10),
            initial=(0 if shardshuffle is None else 100),
        )
        .map(postprocess_fn)
        .with_length(dataset_length)
    )
    
    return dataset


class ObjaverseDataLoader(LightningDataModule):

    color_background = [255.0, 255.0, 255.0, 255.0]

    def __init__(
        self,
        train_config: DictConfig,
        val_config: DictConfig,
        test_config: DictConfig | None,
        batch_size: int,
        num_workers: int,
        load_all_views: bool = False,
    ):
        super(ObjaverseDataLoader, self).__init__()
        if test_config is None:
            test_config = val_config

        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config

        self.transform = ToTensor()

        self.train_postprocess_fn = partial(self.postprocess_fn, config=train_config)
        self.val_postprocess_fn = partial(self.postprocess_fn, config=val_config)
        self.test_postprocess_fn = partial(self.postprocess_fn, config=test_config)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_all_views = load_all_views


    def setup(self, stage: str) -> None:        

        # Adjusting the length of an epoch for multi-node training
        world_size = 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            try:
                import torch.distributed

                if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                ):
                    group = torch.distributed.group.WORLD
                    world_size = torch.distributed.get_world_size(group=group)
            except ModuleNotFoundError:
                pass

        if stage == "fit" or stage is None:
            self.train_dataset = create_dataset(self.train_config.urls, self.train_config.length, self.batch_size,
                                                True, "train", self.train_postprocess_fn, world_size)
            self.val_dataset = create_dataset(self.val_config.urls, self.val_config.length, self.batch_size,
                                              True, "val", self.val_postprocess_fn, world_size)
        
        if stage == "test" or stage is None:
            self.test_dataset = create_dataset(self.test_config.urls, self.test_config.length, self.batch_size,
                                               False, "val", self.test_postprocess_fn, world_size, force_no_shuffle=True)



    def postprocess_fn(self, sample, config):

        num_viewpoints = config["num_viewpoints"]

        try:

            image_list = [obj for obj in sample.keys() if "image" in obj]

            idxs = random.sample(range(len(image_list)), num_viewpoints+1)

            points = sample["points.npy"]
            focals = sample["focals.npy"]
            poses = sample["poses.npy"]

            images = []

            for i in idxs:
                image_key = image_list[i]
                images.append(self.transform(sample[image_key]))

            images = torch.stack(images)
            output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs])

            return output

        except:

            return None

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
