#!/bin/bash
# export OMP_NUM_THREADS=1

ConfigPath=config/config_MOS.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
NumGPUs=4

python3 -m torch.distributed.run \
    --nproc_per_node=$NumGPUs GNU_MOS_train.py \
    --config $ConfigPath \
    --start_validating_epoch 0 \
    --keep_training