#!/bin/bash
export OMP_NUM_THREADS=8

ConfigPath=config/config_MOS.py

export CUDA_VISIBLE_DEVICES=0,1
NumGPUs=2

python3 -m torch.distributed.run \
    --nproc_per_node=$NumGPUs train.py \
    --config $ConfigPath \
    --start_validating_epoch 0
    # --keep_training