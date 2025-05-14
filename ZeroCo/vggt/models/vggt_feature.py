# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from models.croco.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
import torch.nn.functional as F
import numpy as np

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.x_normal = np.linspace(-1,1,37)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,37)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum
    
    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        feature: bool = False,
        layer: int = None
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        # chunk = 1374
        # feature = aggregated_tokens_list[layer].squeeze(0)[:, :chunk, :]
        # #feature = self.aggregator.frame_blocks[layer].attn.saved_feature[:, :chunk, :1024]
        # feat1 = feature[0, patch_start_idx:].unsqueeze(0) # [1, 1369, 1024]
        # feat2 = feature[1, patch_start_idx:].unsqueeze(0) # [1, 1369, 1024]
        # l2norm = FeatureL2Norm()

        # # from (b, n, d) normalize along the d dimension.
        # feat1 = l2norm(feat1, dim=2)  # normalize along descriptor dimension
        # feat2 = l2norm(feat2, dim=2)
        # corr = torch.einsum('bnd, bmd -> bnm', feat1, feat2)   # torch.Size([1, 196, 196])
        
        # beta=1e-4

        # grid_x, grid_y = self.soft_argmax(corr.transpose(-1,-2).view(1, -1, 37, 37), beta=beta)
        # coarse_flow = torch.cat((grid_x, grid_y), dim=1)    
        # flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow) 

        # feature_size = 37
        # flow_est = F.interpolate(flow_est, size=(518, 518), mode='bilinear', align_corners=False) 
        # flow_est[:,0,:,:] *= 518/feature_size 
        # flow_est[:,1,:,:] *= 518/feature_size 
        
        aggregated_tokens_list1, patch_start_idx = self.aggregator(images[:, :1])
        aggregated_tokens_list2, patch_start_idx = self.aggregator(images[:, 1:])
        
        chunk = 1374
        feature1 = aggregated_tokens_list1[layer].squeeze(0)[:, :chunk, 1024:]
        feature2 = aggregated_tokens_list2[layer].squeeze(0)[:, :chunk, 1024:]
        #feature = self.aggregator.frame_blocks[layer].attn.saved_feature[:, :chunk, :1024]

        feat1 = feature1[:, patch_start_idx:] # [1, 1369, 1024]
        feat2 = feature2[:, patch_start_idx:] # [1, 1369, 1024]
        l2norm = FeatureL2Norm()

        # from (b, n, d) normalize along the d dimension.
        feat1 = l2norm(feat1, dim=2)  # normalize along descriptor dimension
        feat2 = l2norm(feat2, dim=2)
        corr = torch.einsum('bnd, bmd -> bnm', feat1, feat2)   # torch.Size([1, 196, 196])
        
        beta=1e-4

        grid_x, grid_y = self.soft_argmax(corr.transpose(-1,-2).view(1, -1, 37, 37), beta=beta)
        coarse_flow = torch.cat((grid_x, grid_y), dim=1)    
        flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow) 

        feature_size = 37
        flow_est = F.interpolate(flow_est, size=(518, 518), mode='bilinear', align_corners=False) 
        flow_est[:,0,:,:] *= 518/feature_size 
        flow_est[:,1,:,:] *= 518/feature_size 
        
        return flow_est

        # input_num = images.shape[1]
        # attn_map_all = [[] for i in range(input_num)]
        # for img_idx in range(input_num):
        #     for i in [4, 11, 17, 23]:
        #         feature = aggregated_tokens_list[i].squeeze(0)[img_idx] # [1374, 2048]
        #         feature = feature[patch_start_idx:, 1024:] # [1369, 1048] - global

        #         attn_map_all[img_idx].append(feature) # [1369, 1048]
                # attn_map3_all.append(feature3)
        # for i in [4, 11, 17, 23]: 
        #     import pdb; pdb.set_trace()
        #     query = self.aggregator.global_blocks[i].attn.saved_q[:, :, :chunk] # [1, 16, 261, 64]
        #     k = self.aggregator.global_blocks[i].attn.saved_k # [1, 16, 783, 64]
        #     attn = query @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(query.shape[-1])) # [1, 16, 261, 783])
        #     attn = attn.softmax(dim=-1)
            
        #     attn_map2 = attn[:, :, :, chunk:chunk*2] # [1, 16, 261, 261]
        #     attn_map3 = attn[:, :, :, chunk*2:]
            
        #     attn_map2 = attn_map2[:, :, patch_start_idx:, patch_start_idx:]
        #     attn_map3 = attn_map3[:, :, patch_start_idx:, patch_start_idx:]
            
        #     attn_map2_all.append(attn_map2)
        #     attn_map3_all.append(attn_map3)
            
        
        # for i in [4, 11, 17, 23]: 
        #     #attn_map1 = self.aggregator.frame_blocks[i].attn.saved_attn_map[:, :, :chunk*2, :chunk*2]
        #     attn_map2 = self.aggregator.frame_blocks[i].attn.saved_attn_map[1, :, patch_start_idx:, patch_start_idx:]
        #     attn_map3 = self.aggregator.frame_blocks[i].attn.saved_attn_map[2, :, patch_start_idx:, patch_start_idx:]
        #     attn_map2_all.append(attn_map2)
        #     attn_map3_all.append(attn_map3)
        
        return attn_map_all
        if feature == True:
            dpt_layer = [4, 11, 17, 23]
            mid_layer_features = []
        
            for layer_idx in dpt_layer:
                x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]  # shape: [B, C, P]
                mid_layer_features.append(x)

            mid_layer_features = torch.cat(mid_layer_features, dim=0)
                
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

                if self.depth_head is not None:
                    depth, depth_conf, depth_feat = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf
                    predictions['depth_feature'] = depth_feat

                if self.point_head is not None:
                    pts3d, pts3d_conf, pts3d_feat = self.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["world_points"] = pts3d
                    predictions["world_points_conf"] = pts3d_conf
                    predictions['world_points_feature'] = pts3d_feat
                    

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf

            predictions["images"] = images

            return mid_layer_features, predictions
        
        else:
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            
            return predictions
