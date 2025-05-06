#!/bin/bash

ConfigPath=config/config_MOS.py

export CUDA_VISIBLE_DEVICES=0
python test_speed.py --config $ConfigPath
                                