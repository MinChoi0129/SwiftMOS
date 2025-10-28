#!/bin/bash

ConfigPath=config/config_MOS.py

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
NumGPUs=1

python3 -m torch.distributed.run \
    --nproc_per_node=$NumGPUs SwiftMOS_train.py \
    --config $ConfigPath \
    --start_validating_epoch 4
    # --keep_training # (if you want to continue training, check pretrain_epoch in config/config_MOS.py)