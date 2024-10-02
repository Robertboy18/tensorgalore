for rank in 8 16 32 64 128
do
    CUDA_VISIBLE_DEVICES=1 python train_ns_galore.py --opt.galore_rank=$rank --opt.first_dim_rollup=1 --opt.naive_galore=True & \\
    CUDA_VISIBLE_DEVICES=2 python train_ns_galore.py --opt.galore_rank=$rank --opt.first_dim_rollup=2 --opt.naive_galore=True & \\ 
    CUDA_VISIBLE_DEVICES=3 python train_ns_galore.py --opt.galore_rank=$rank --opt.first_dim_rollup=3 --opt.naive_galore=True
done
