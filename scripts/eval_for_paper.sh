#!/bin/bash

DatasetPath=/home/workspace/KITTI/dataset

PredictionsPath=experiments/config_TripleMOS/results
DataConfig=datasets/semantic-kitti-mos.yaml

python semantic-kitti-api/evaluate_mos.py \
                                -d $DatasetPath \
                                -p $PredictionsPath \
                                -dc $DataConfig \
                                -s valid 