#!/bin/bash

ConfigPath=config/config_TripleMOS.py

export CUDA_VISIBLE_DEVICES=1
python /home/work/TripleMOS/test_speed.py --config $ConfigPath
                                