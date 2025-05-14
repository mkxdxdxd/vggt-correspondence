from os.path import join
from typing import Union, Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import inspect
import os
import sys

from omegaconf import OmegaConf, DictConfig
from jaxtyping import Float

import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from torchvision.utils import save_image

from .models import (
    PoseGuider,
    UNet2DConditionModel,
    UNet3DConditionModel,
    ReferenceAttentionControl
)

from .utils import (
    reprojector,
    ndc_rasterizer,
    one_to_one_rasterizer
)

from .ops import get_viewport_matrix

# from ..train_model import prepare_duster_embedding

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from .dust3r.utils.geometry import find_reciprocal_matches, xy_grid

class GenWarp():
    @dataclass
    class Config():
        version: str = 'naive'
        warped_geo: bool = False 
        pretrained_model_path: str = ''
        checkpoint_name: str = ''
        half_precision_weights: bool = False
        height: int = 512
        width: int = 512
        num_inference_steps: int = 20
        guidance_scale: float = 3.5
        num_references: int = 3
        embedder_input_dim : int = 2
        depth_condition: bool = False
        use_mesh: bool = False
        use_normal: bool = False
        use_plucker: bool = False
        training_val: bool = False
        multitask: bool = False
        ref_expand: bool = False

    cfg: Config

    class Embedder():
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.create_embedding_fn()

        def create_embedding_fn(self) -> None:
            embed_fns = []
            d = self.kwargs['input_dims']
            out_dim = 0
            if self.kwargs['include_input']:
                embed_fns.append(lambda x : x) # 2
                out_dim += d

            max_freq = self.kwargs['max_freq_log2']
            N_freqs = self.kwargs['num_freqs']

            if self.kwargs['log_sampling']:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d
            
            # import pdb; pdb.set_trace()

            self.embed_fns = embed_fns
            self.out_dim = out_dim

        def embed(self, inputs) -> Tensor:
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def __init__(
        self,
        cfg: Optional[Union[dict, DictConfig]] = None,
        device: Optional[str] = 'cuda:0',
        reference_unet = None,
        denoising_unet = None,
        pose_guider = None,
        scheduler = None,
        ref_reader=None,
        ref_writer=None
    ) -> None:
        self.cfg = OmegaConf.structured(self.Config(**cfg))
        self.model_path = join(
            self.cfg.pretrained_model_path, self.cfg.checkpoint_name
        )
        self.device = device

        if self.cfg.training_val:
            self.reference_unet=reference_unet
            self.denoising_unet=denoising_unet
            self.pose_guider=pose_guider
        
        self.configure()
        
    def match_scale_shift(self, tgt_geo, synthesized, full):
        """
        Aligns the scale and shift of a source point cloud to match a target point cloud.

        Args:
            source_pc (torch.Tensor): Source point cloud (N, 3), incorrectly scaled & shifted.
            target_pc (torch.Tensor): Target point cloud (N, 3), correctly scaled.

        Returns:
            torch.Tensor: Aligned source point cloud (N, 3).
            float: Estimated scale factor.
            torch.Tensor: Estimated translation vector (1, 3).
        """
        # Compute centroids
        
        
        
        target_pc = tgt_geo.permute(0,2,3,1).reshape(-1,3)
        source_pc = synthesized.permute(0,2,3,1).reshape(-1,3)
        
        nonzero_idx = torch.nonzero(target_pc[:,0])[...,0]
        
        target_pc = target_pc[nonzero_idx]
        source_pc = source_pc[nonzero_idx]
        
        centroid_s = source_pc.mean(dim=0, keepdim=True)
        centroid_t = target_pc.mean(dim=0, keepdim=True)

        # Center the point clouds
        source_centered = source_pc - centroid_s
        target_centered = target_pc - centroid_t

        # Compute scale factor (using least squares)
        # scale = (target_centered.norm(dim=1).mean()) / (source_centered.norm(dim=1).mean())
        scale = (torch.abs(target_centered).mean(dim=0)) / (torch.abs(source_centered).mean(dim=0))

        # Compute translation vector
        translation = centroid_t - scale[None,...] * centroid_s

        # Apply transformation to source point cloud
        # import pdb; pdb.set_trace()
        aligned_source_pc = scale[None,...,None,None] * full + translation[...,None,None]

        return aligned_source_pc, scale, translation

    def configure(self) -> None:
        print(f"Loading GenWarp...")

        # Configurations.
        self.dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        self.viewport_mtx: Float[Tensor, 'B 4 4'] = get_viewport_matrix(
            self.cfg.width, self.cfg.height,
            batch_size=1, device=self.device
        ).to(self.dtype)

        # Load models.
        if not self.cfg.training_val:
            self.load_models()
        else:
            self.load_val_models()

        # Timestep
        self.scheduler.set_timesteps(
            self.cfg.num_inference_steps, device=self.device)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        print(f"Loaded GenWarp.")

    def load_models(self) -> None:
        
        if self.cfg.version == "marigold":
            self.vae = AutoencoderKL.from_pretrained(
                '/media/Marigold/checkpoint/marigold-v1-0/vae'
            ).to(self.device, dtype=self.dtype)
        
            # Image encoder.
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
            ).to(self.device, dtype=self.dtype)
            
        else:
            # VAE.
            self.vae = AutoencoderKL.from_pretrained(
                join(self.cfg.pretrained_model_path, 'sd-vae-ft-mse')
            ).to(self.device, dtype=self.dtype)
        
            # Image encoder.
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                join(self.cfg.pretrained_model_path, 'image_encoder')
            ).to(self.device, dtype=self.dtype)

        # Image processor.
        self.vae_scale_factor = \
            2 ** (len(self.vae.config.block_out_channels) - 1)
            
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        
        self.clip_image_processor = CLIPImageProcessor()



        # Reference Unet.
        self.reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                # join(self.model_path, 'config_ref.json')
                join(self.model_path, 'config.json')
        )).to(self.device, dtype=self.dtype)
        self.reference_unet.load_state_dict(torch.load(
            join(self.model_path, 'reference_unet.pth'),
            map_location= 'cpu'),
        )

        # Denoising Unet.
        # if self.cfg.multitask: 
        
        if self.cfg.version == "switcher":
            self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                join(self.model_path, 'denoising_config.json'),
                join(self.model_path, 'denoising_unet.pth')
            ).to(self.device, dtype=self.dtype)
            
        else:
            try:
                self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                    join(self.model_path, 'denoising_config.json'),
                    join(self.model_path, 'denoising_unet.pth')
                ).to(self.device, dtype=self.dtype)
            #
            except:
                self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                    join(self.model_path, 'config.json'),
                    join(self.model_path, 'denoising_unet.pth')
                ).to(self.device, dtype=self.dtype)

        self.unet_in_channels = self.denoising_unet.config.in_channels

        if self.cfg.multitask:
            self.unet_in_channels = 8

        # Pose guider.

        channel_dep = self.cfg.embedder_input_dim * 5 + 1

        if self.cfg.depth_condition:
            channel_dep += 1
        if self.cfg.use_normal:
            channel_dep += 3
        if self.cfg.use_plucker:
            channel_dep += 6

        self.pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            conditioning_channels=channel_dep,
        ).to(self.device, dtype=self.dtype)

        # try:
        self.pose_guider.load_state_dict(torch.load(
            join(self.model_path, 'pose_guider.pth'),
            map_location='cpu'),
        )
        # except:
        #     pass
        # Noise scheduler
        sched_kwargs = OmegaConf.to_container(OmegaConf.create({
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'steps_offset': 1,
            'clip_sample': False
        }))
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing='trailing',
            prediction_type='v_prediction',
        )
        self.scheduler = DDIMScheduler(**sched_kwargs)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)
        self.pose_guider.requires_grad_(False)

        self.depth_condition = False

        # Coordinates embedding.
        # self.embedder = self.get_embedder(1)
        # import pdb; pdb.set_trace()s
        self.embedder = self.get_embedder(2, input_dim=self.cfg.embedder_input_dim)

    
    def load_val_models(self):
        self.vae = AutoencoderKL.from_pretrained(
            join(self.cfg.pretrained_model_path, 'sd-vae-ft-mse')
        ).to(self.device, dtype=self.dtype)

        # Image processor.
        self.vae_scale_factor = \
            2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()

        # Image encoder.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            join(self.cfg.pretrained_model_path, 'image_encoder')
        ).to(self.device, dtype=self.dtype)

        sched_kwargs = OmegaConf.to_container(OmegaConf.create({
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'steps_offset': 1,
            'clip_sample': False
        }))

        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing='trailing',
            prediction_type='v_prediction',
        )
        self.scheduler = DDIMScheduler(**sched_kwargs)

        self.unet_in_channels = self.denoising_unet.config.in_channels

        if self.cfg.multitask:
            self.unet_in_channels = 8

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)
        self.pose_guider.requires_grad_(False)

        self.depth_condition = False

        # Coordinates embedding.
        # self.embedder = self.get_embedder(1)
        # import pdb; pdb.set_trace()s
        self.embedder = self.get_embedder(2, input_dim=self.cfg.embedder_input_dim)


    def get_embedder(self, multires, input_dim=3):
        embed_kwargs = {
            'include_input' : True,
            'input_dims' : input_dim,
            'max_freq_log2' : multires-1,
            'num_freqs' : multires,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],
        }

        embedder_obj = self.Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        return embed

    def __call__(
        self,
        images,
        conditions,
        renders,
        tgt_gt = None,
        tgt_geo = None,
        ref_geo = None,
    ) -> Dict[str, Tensor]:
        """ Perform NVS.
        """
        src_images = [img[0] for img in images["ref"]]
        view_size = src_images[0].shape[0]
        
        warped_image = renders["warped"]
        # warped_image = warped_image * mesh_normal_mask.permute(0,3,1,2)
        
        # import pdb; pdb.set_trace()
        
        if self.cfg.version == "switcher":
            conditions["switcher_stat"] = "image"
        
        # NVS.
        latents_clean = self.perform_nvs(
            src_images=src_images,
            tgt_warped=warped_image,
            ref_geo=ref_geo,
            tgt_geo=tgt_geo,
            **conditions
        )
        
        # latents_clean = self.perform_nvs(src_images=src_images, tgt_warped=warped_image, ref_geo=ref_geo, tgt_geo=tgt_geo, **conditions)

        if latents_clean.shape[1] != 4:
            loc_latents = latents_clean[:,4:]
            latents_clean = latents_clean[:,:4]

        # Decode to images.
        synthesized = self.decode_latents(latents_clean, normalize=True)
        
        result = synthesized
        
        # Scale-shift matching
        # mask = (tgt_geo != 0).float()
        
        # result, scale, shift = self.match_scale_shift(tgt_geo[...,448:,448:], synthesized[...,448:,448:], synthesized)
        # print(scale)
        # print(shift)
        
        # import pdb; pdb.set_trace()
        
        # save_image(renders["warped"],"praise_0.png")
        # save_image(tgt_gt * 0.5 + 0.5,"praise_1.png")
        # save_image(synthesized * 0.5 + 0.5,"praise_2.png")
        
        ptsmap_min = torch.tensor([-0.7, -0.7, 0.01])
        ptsmap_max = torch.tensor([0.7, 0.3, 1.5])
        
        # import pdb; pdb.set_trace()

        if self.cfg.multitask:
            other = self.decode_latents(loc_latents)
        else:
            other = None
            # import pdb; pdb.set_trace()

        inference_out = {
            'synthesized': result,
            'warped': renders['warped'],
            # 'correspondence': renders['correspondence'],
            'tgt_depth': renders["tgt_depth"],
            # 'ref_depth': renders["ref_depth"],
            # 'other' : other
        }

        return inference_out

    def preprocess_image(
        self,
        image: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        image = F.interpolate(
            image, (self.cfg.height, self.cfg.width)
        )
        return image

    def get_image_prompt(
        self,
        src_image: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, '2 B L']:
        ref_image_for_clip = self.vae_image_processor.preprocess(
            src_image, height=224, width=224
        )
        ref_image_for_clip = ref_image_for_clip * 0.5 + 0.5

        clip_image = self.clip_image_processor.preprocess(
            ref_image_for_clip, do_rescale=False, return_tensors='pt'
        ).pixel_values

        clip_image_embeds = self.image_encoder(
            clip_image.to(self.device, dtype=self.image_encoder.dtype)
        ).image_embeds

        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        image_prompt_embeds = torch.cat(
            [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
        )

        return image_prompt_embeds

    def encode_images(
        self,
        rgb: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        rgb = self.vae_image_processor.preprocess(rgb)
        latents = self.vae.encode(rgb).latent_dist.mean
        latents = latents * 0.18215
        return latents

    def decode_latents(
        self,
        latents: Float[Tensor, 'B C H W'],
        normalize=True
    ) -> Float[Tensor, 'B C H W']:
        latents = 1 / 0.18215 * latents
        rgb = []
        for frame_idx in range(latents.shape[0]):
            rgb.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)

        rgb = torch.cat(rgb)
        
        if normalize:
            rgb = (rgb / 2 + 0.5).clamp(0, 1)
        return rgb.squeeze(2)

    def get_reference_controls(
        self,
        batch_size: int
    ) -> Tuple[ReferenceAttentionControl, ReferenceAttentionControl]:
        reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode='read',
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing',
            num_references=self.cfg.num_references
        )
        
        writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode='write',
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing',
            num_references=self.cfg.num_references
        )

        return reader, writer

    def prepare_extra_step_kwargs(
        self,
        generator,
        eta
    ) -> Dict[str, Any]:
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def get_pose_features(
        self,
        src_embed: Float[Tensor, 'B C H W'],
        trg_embed: Float[Tensor, 'B C H W'],
        do_classifier_guidance: bool = True
    ) -> Tuple[Tensor, Tensor]:
        pose_cond_tensor = src_embed.unsqueeze(2)
        pose_cond_tensor = pose_cond_tensor.to(
            device=self.device, dtype=self.pose_guider.dtype
        )
        pose_cond_tensor_2 = trg_embed.unsqueeze(2)
        pose_cond_tensor_2 = pose_cond_tensor_2.to(
            device=self.device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea_2 = self.pose_guider(pose_cond_tensor_2)

        if do_classifier_guidance:
            pose_fea = torch.cat([pose_fea] * 2)
            pose_fea_2 = torch.cat([pose_fea_2] * 2)

        return pose_fea, pose_fea_2

    def perform_nvs(
        self,
        src_images,
        tgt_warped,
        ref_geo,
        tgt_geo,
        ref_embeds,
        tgt_embed,
        ref_correspondence,
        switcher_stat = "image",
        eta: float=0.0,
        correspondence = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        **kwargs,
    ) -> Float[Tensor, 'B C H W']:
        batch_size = src_images[0].shape[0]

        reference_control_reader, reference_control_writer = \
            self.get_reference_controls(batch_size)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(
            generator, eta
        )

        src_embed_list = ref_embeds

        with torch.no_grad():
            # Create fake inputs. It'll be replaced by pure noise.
            for i, src_image in enumerate(src_images):
                
                if self.cfg.version == "naive":
                    latents = torch.randn(
                        batch_size,
                        self.unet_in_channels,
                        self.cfg.height // self.vae_scale_factor,
                        self.cfg.width // self.vae_scale_factor
                    ).to(self.device, dtype=src_image.dtype)
                
                elif self.cfg.version == "marigold":
                    latents = torch.randn(
                        batch_size,
                        4,
                        self.cfg.height // self.vae_scale_factor,
                        self.cfg.width // self.vae_scale_factor
                    ).to(self.device, dtype=src_image.dtype)

                initial_t = torch.tensor(
                    [self.num_train_timesteps - 1] * batch_size
                ).to(self.device, dtype=torch.long)

                # Add noise.
                noise = torch.randn_like(latents)
                latents_noisy_start = self.scheduler.add_noise(
                    latents, noise, initial_t
                )
                latents_noisy_start = latents_noisy_start.unsqueeze(2)
                
                # Prepare clip image embeds.
                image_prompt_embeds = self.get_image_prompt(src_image)

                # Prepare ref image latents.
                ref_image_latents = self.encode_images(src_image)

                # if self.cfg.ref_expand:
                #     src_cor = ref_correspondence[i][None,...].permute(0,3,1,2)
                #     ref_cor_latents = self.encode_images(src_cor)

                #     ref_image_latents = torch.cat((ref_image_latents,ref_cor_latents), dim=1)

                # Prepare pose condition image.
                # import pdb; pdb.set_trace()

                embed_dim = self.cfg.embedder_input_dim * 5 + 1 

                if self.depth_condition:
                    embed_dim += 1
                
                pose_fea, pose_fea_2 = self.get_pose_features(
                    src_embed_list[i], tgt_embed
                )

                pose_fea = pose_fea[:, :, 0, ...]
                
                if self.cfg.version == "marigold":
                    ref_depth_latents = self.encode_images(ref_geo[i][None,...])
                    ref_image_latents = torch.cat((ref_image_latents, ref_depth_latents), dim=1)
                    
                # Forward reference images.
                self.reference_unet(
                    ref_image_latents.repeat(2, 1, 1, 1),
                    torch.zeros(batch_size * 2).to(ref_image_latents),
                    encoder_hidden_states=image_prompt_embeds,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                    reference_idx=i,
                )
            
            # Update the denosing net with reference features.
            reference_control_reader.update(
                reference_control_writer,
                correspondence=correspondence
            )

            timesteps = self.scheduler.timesteps
            latents_noisy = latents_noisy_start
            
            task_emb = None
                        
            if self.cfg.version == "marigold":
                warped_latent = self.encode_images(tgt_warped).unsqueeze(2)
                warped_latent = torch.cat([warped_latent]*2)
                
                if tgt_geo != None:
                    warped_geo = self.encode_images(tgt_geo).unsqueeze(2)
                    warped_geo = torch.cat([warped_geo]*2)
                                
            elif self.cfg.version == "switcher":
                warped_geo = self.encode_images(tgt_geo).unsqueeze(2)
                warped_geo = torch.cat([warped_geo]*2)
                
                if switcher_stat == "image":
                    task_emb = torch.tensor([0, 1]).float().unsqueeze(0).repeat(1, 1).to(latent_model_input.device)
                elif switcher_stat == "geometry":
                    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(latent_model_input.device)
                
            for t in timesteps:
                # Prepare latents.
                
                latent_model_input = torch.cat([latents_noisy] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                if self.cfg.version == "marigold":
                    # If marigold, concat the image latent
                    latent_model_input = torch.cat((warped_latent, latent_model_input), dim=1)
                    
                    if tgt_geo != None:
                        latent_model_input = torch.cat((latent_model_input, warped_geo), dim=1)
                        
                elif self.cfg.version == "switcher":
                    latent_model_input = torch.cat((latent_model_input, warped_geo), dim=1)

                # Denoise.
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    pose_cond_fea=pose_fea_2,
                    return_dict=False,
                    class_labels=task_emb
                )[0]

                # CFG.
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # t -> t-1
                latents_noisy = self.scheduler.step(
                    noise_pred, t, latents_noisy, **extra_step_kwargs,
                    return_dict=False
                )[0]

            # Noise disappears eventually
            latents_clean = latents_noisy

        reference_control_reader.clear()
        reference_control_writer.clear()

        return latents_clean.squeeze(2)