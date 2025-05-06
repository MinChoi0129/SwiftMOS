#!/bin/bash
export OMP_NUM_THREADS=4

ConfigPath=config/config_MOS.py

export CUDA_VISIBLE_DEVICES=0
NumGPUs=1

python3 -m torch.distributed.run \
    --nproc_per_node=$NumGPUs train.py \
    --config $ConfigPath \
    --start_validating_epoch 0