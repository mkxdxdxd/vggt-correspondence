CUDA_VISIBLE_DEVICES=0 python preprocess_co3d.py --gpu_num=8 &
CUDA_VISIBLE_DEVICES=1 python preprocess_co3d.py --gpu_num=9 &
CUDA_VISIBLE_DEVICES=2 python preprocess_co3d.py --gpu_num=10 &
CUDA_VISIBLE_DEVICES=3 python preprocess_co3d.py --gpu_num=11 &
CUDA_VISIBLE_DEVICES=4 python preprocess_co3d.py --gpu_num=12 &
CUDA_VISIBLE_DEVICES=5 python preprocess_co3d.py --gpu_num=13 &
CUDA_VISIBLE_DEVICES=6 python preprocess_co3d.py --gpu_num=14 &
CUDA_VISIBLE_DEVICES=7 python preprocess_co3d.py --gpu_num=15 &

wait

echo "Both processes have completed."