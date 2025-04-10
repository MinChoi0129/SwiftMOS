#!/bin/bash

DatasetPath=/home/ssd_4tb/minjae/KITTI/dataset
PredictionsPath=/home/work/SMVF/experiments/config_TripleMOS/results
DataConfig=/home/work/SMVF/datasets/semantic-kitti-mos.yaml

python /home/work/semantic-kitti-api/evaluate_mos.py \
                                -d $DatasetPath \
                                -p $PredictionsPath \
                                -dc $DataConfig
                                -s valid \