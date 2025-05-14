CUDA_VISIBLE_DEVICES=0 python preprocess_co3d.py --gpu_num=16 &
CUDA_VISIBLE_DEVICES=1 python preprocess_co3d.py --gpu_num=17 &
CUDA_VISIBLE_DEVICES=2 python preprocess_co3d.py --gpu_num=18 &
CUDA_VISIBLE_DEVICES=3 python preprocess_co3d.py --gpu_num=19 &
CUDA_VISIBLE_DEVICES=4 python preprocess_co3d.py --gpu_num=20 &
CUDA_VISIBLE_DEVICES=5 python preprocess_co3d.py --gpu_num=21 &
CUDA_VISIBLE_DEVICES=6 python preprocess_co3d.py --gpu_num=22 &
CUDA_VISIBLE_DEVICES=7 python preprocess_co3d.py --gpu_num=23 &

wait

echo "Both processes have completed."