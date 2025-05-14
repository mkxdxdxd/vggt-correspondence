# CUDA_VISIBLE_DEVICES=1 python eval_hpatches.py \
#     --hpatches_path /workspace/minkyung/dift/hpatches-sequences-release \
#     --kpts_path ./superpoint-1k \
#     --save_path ./hpatches_results_700 \
#     --dift_model vggt \
#     --layer 23 \
#     --img_size 700 700 \
#     # --t 261 \
#     # --up_ft_index 1 \
#     # --ensemble_size 8


CUDA_VISIBLE_DEVICES=1 python eval_homography.py \
    --hpatches_path /workspace/minkyung/dift/hpatches-sequences-release \
    --save_path ./hpatches_results_700 \
    --feat dift_vggt \
    --metric cosine \
    --mode lmeds