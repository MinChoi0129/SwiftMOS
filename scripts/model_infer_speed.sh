#!/bin/bash

ConfigPath=config/config_MOS.py
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
python SwiftMOS_test_speed.py --config $ConfigPath
