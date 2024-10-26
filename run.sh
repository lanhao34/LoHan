#!/bin/bash

## for single GPU
python main.py \
    --hidden_dim 5120 \
    --num_heads 80 \
    --num_layers  40 \
    --batch_size  64 \
    --is_swap_and_recompute  0 \
    --is_swap_prior  1 \
    --is_fully_swap  0 \
    --swap_ratio 0.3 \
    --is_new_param_async  1 \
    --is_grad_async  1 \
    --is_mp  1 \
    --is_nvme  1 \
    --is_nvme_async  1 \
    --is_nvme_rearrange  1 \
    --sb_config ./config.json

## for multi-GPU
## use CUDA_VISIBLE_DEVICES to set the GPU to be used
# CUDA_VISIBLE_DEVICES=0,1 \
#     torchrun main.py \
#     --hidden_dim 5120 \
#     --num_heads 80 \
#     --num_layers  40 \
#     --batch_size  64 \
#     --is_swap_and_recompute  0 \
#     --is_swap_prior  1 \
#     --is_fully_swap  0 \
#     --swap_ratio 0.8 \
#     --is_new_param_async  1 \
#     --is_grad_async  1 \
#     --is_mp  1 \
#     --is_nvme  1 \
#     --is_nvme_async  1 \
#     --is_nvme_rearrange  1 \
#     --sb_config /home/xiejun/Ratel_Private/config.json