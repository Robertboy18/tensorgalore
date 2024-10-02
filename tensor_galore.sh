for rank in 8 16 32 64 128
do
    CUDA_VISIBLE_DEVICES=1 python train_ns_galore.py --opt.galore_rank=$rank
done
