#!/bin/bash

DatasetPath=/home/workspace/KITTI/dataset

PredictionsPath=/home/workspace/work/TripleMOS/experiments/config_TripleMOS/config_TripleMOS/results
DataConfig=/home/workspace/work/TripleMOS/datasets/semantic-kitti-mos.yaml

python semantic-kitti-api/evaluate_mos.py \
                                -d $DatasetPath \
                                -p $PredictionsPath \
                                -dc $DataConfig \
                                -s valid 