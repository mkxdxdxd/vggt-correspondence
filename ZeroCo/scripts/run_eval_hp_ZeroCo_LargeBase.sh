#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --model croco \
    --croco_ckpt /workspace/minkyung/dift/croco/checkpoint/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --save_dir ./vis/eval/hp/ZeroCo_LargeBase \
    --log_warped_images \
