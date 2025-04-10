#!/bin/bash

ConfigPath=config/config_TripleMOS.py
CheckpointModelEpoch_from=11
CheckpointModelEpoch_to=11

export CUDA_VISIBLE_DEVICES=1,2
NumGPUs=2 # should be the 'length of cuda visible devices'

python3 -m torch.distributed.launch --nproc_per_node=$NumGPUs evaluate.py \
    --config $ConfigPath \
    --start_epoch $CheckpointModelEpoch_from \
    --end_epoch $CheckpointModelEpoch_to \
    --save_label