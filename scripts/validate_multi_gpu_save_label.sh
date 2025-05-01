#!/bin/bash

ConfigPath=config/config_TripleMOS.py
CheckpointModelEpoch=39

export CUDA_VISIBLE_DEVICES=1

python3 evaluate.py \
    --config $ConfigPath \
    --model_epoch $CheckpointModelEpoch \
    --save_label