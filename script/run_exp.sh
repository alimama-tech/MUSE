#!/bin/bash

# change gpu numbers within [1,2,4,8]
GPU_LIST="0,1,2,3,4,5,6,7"
NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)

EXP_NAME="1k_top50_muse_reproduce_dim_16_sparse_lr_2e-3_id_p90_scl_p90_scl@gpu_v1"
LOG_NAME="./logs/${EXP_NAME}.log"

# do not delete --use_ddp
CUDA_VISIBLE_DEVICES="$GPU_LIST" nohup torchrun --nproc_per_node=$NUM_GPUS main.py \
    --config config/muse.json \
    --use_ddp \
    --exp_name $EXP_NAME \
    > $LOG_NAME 2>&1 &