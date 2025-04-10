#!/bin/bash

ConfigPath=config/config_TripleMOS.py

export CUDA_VISIBLE_DEVICES=1,2
NumGPUs=2 # should be the 'length of cuda visible devices'

python3 -m torch.distributed.launch \
    --nproc_per_node=$NumGPUs train.py \
    --config $ConfigPath 
    # --keep_training
    # --start_validating_epoch 20