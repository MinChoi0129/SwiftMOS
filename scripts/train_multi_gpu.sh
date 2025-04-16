#!/bin/bash

ConfigPath=config/config_TripleMOS.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
NumGPUs=4 # should be the 'length of cuda visible devices'

python3 -m torch.distributed.run \
    --nproc_per_node=$NumGPUs train.py \
    --config $ConfigPath \
    --start_validating_epoch 0 \
    --keep_training