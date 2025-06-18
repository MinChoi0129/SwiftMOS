#!/bin/bash

ConfigPath=config/config_MOS.py
CheckpointModelEpoch=39
EvalMode=val # or test

export CUDA_VISIBLE_DEVICES=0

python3 GNU_MOS_evaluate.py \
    --config $ConfigPath \
    --model_epoch $CheckpointModelEpoch \
    --eval_mode $EvalMode
    # --save_label