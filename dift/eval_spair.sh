CUDA_VISIBLE_DEVICES=1 python eval_spair.py \
    --dataset_path ./SPair-71k \
    --save_path ./spair_ft_global_4 \
    --layer 4 \
    --dift_model vggt \
    --img_size 518 518 \