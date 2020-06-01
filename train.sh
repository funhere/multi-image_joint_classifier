#!/bin/bash

pip install matplotlib
pip install sklearn
conda install -c pytorch pytorch torchvision cudatoolkit=10.0

##tf_efficientnet_b4
#./distributed_train.sh 1 ./data --model tf_efficientnet_b4 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 1 -j 1 &

## DPN131
./distributed_train.sh 1 ./data --model dpn131 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 1 -j 1 &
