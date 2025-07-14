OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --nnodes=1 --master_port=1111 \
        -m train \
        --conf_path configs/map.json \
        --dataset googlemap \
        --gpu_batch_size 2 \
        --dont_log_wandb
        # --ft \
        # --ft_ckpt workspace/cvpr25-camera-ready/glunet_448x448_occlusionno_cross/latest.pth