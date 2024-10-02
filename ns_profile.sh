#!/bin/bash

source ~/.bashrc;

conda activate tensorgalore;

echo $CONDA_PREFIX;

#for rank in 0.05 0.1 0.25 0.5 0.75 1.0; 

python profile_ns_galore.py \
    --distributed.use_distributed False \
    --fno2d.n_modes_height 120 \
    --fno2d.n_modes_width 120 \
    --fno2d.hidden_channels 256 \
    --fno2d.projection_channels 256 \
    --opt.galore True \
    --opt.naive_galore False \
    --opt.first_dim_rollup 2 \
    --opt.n_epochs 2 \
    --opt.per_layer_opt False \
    --opt.act_checkpoint False \
    --opt.galore_rank '(152,152,72,32)' \
    --data.root /global/homes/d/dhpitt/psc/data/navier_stokes/ \
    --data.train_resolution 128 \
    --data.batch_size 1 \
    --data.test_resolutions [128] \
    --data.test_batch_sizes [1] \
    --data.n_train 8 \
    --data.n_tests [8] \
    --wandb.log False \
    --wandb.name test;

    
