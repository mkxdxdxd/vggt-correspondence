#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --eval_img_size 240 240 \
    --model_img_size 518 518 \
    --model vggt \
    --croco_ckpt /workspace/minkyung/dift/croco/checkpoint/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation enc_feat \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 1e-4 \
    --save_dir ./vis/eval/hp240/ZeroCo_beta1e4 \
    --log_warped_images \
