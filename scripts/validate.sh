#!/bin/bash

ConfigPath=config/config_MOS.py
CheckpointModelEpoch=50
EvalMode=val # val or test

export CUDA_VISIBLE_DEVICES=0

python3 SwiftMOS_evaluate.py \
    --config $ConfigPath \
    --model_epoch $CheckpointModelEpoch \
    --eval_mode $EvalMode \
    --save_label