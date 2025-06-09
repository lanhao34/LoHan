#!/bin/bash
mkdir -p logs
export TORCH_EXTENSIONS_DIR="$PWD/.torch_ext_cache"

cuda_device_names=("A100" "A30")
cuda_device_id=0
export CUDA_VISIBLE_DEVICES=$cuda_device_id
## for single GPU
PROCESS_PATTERN="/home/lh/tools/anaconda3/envs/ratel/bin/python"
PIDS=$(pgrep -f "$PROCESS_PATTERN")
if [ -n "$PIDS" ]; then
  sudo kill -9 $PIDS
fi
## for single GPU
batch_size=64
for model_size in 3B 7B 13B 30B; do
    sudo rm -rf /mnt/cyp/tmp1/zero_stage_3
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/run_${cuda_device_names[$CUDA_VISIBLE_DEVICES]}_${model_size}_LoHan_${batch_size}_${TIMESTAMP}.log"
    sudo -E LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/lh/tools/anaconda3/envs/ratel/bin/python -u main.py \
        --model_size $model_size \
        --batch_size  $batch_size \
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
        --sb_config ./config.json 2>&1 | stdbuf -oL -eL tee -a "$LOG_FILE"
done

model_size=13B
for batch_size in 32 96 128; do
    sudo rm -rf /mnt/cyp/tmp1/zero_stage_3
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/run_${cuda_device_names[$CUDA_VISIBLE_DEVICES]}_${model_size}_LoHan_${batch_size}_${TIMESTAMP}.log"
   sudo -E LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/lh/tools/anaconda3/envs/ratel/bin/python -u main.py \
        --model_size $model_size \
        --batch_size  $batch_size \
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
        --sb_config ./config.json 2>&1 | stdbuf -oL -eL tee -a "$LOG_FILE"
done

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