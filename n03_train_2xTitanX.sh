#!/usr/bin/env bash

PYTHON_BIN=$HOME/anaconda3/bin/python
BATCH=48
CHECK=300
MODEL=resnext-101
VERSN=rnx101-2
GPUS=0,1


for i in {0..9}
do
 $PYTHON_BIN n03_train_conf.py --nfold $i --batch $BATCH --lrate 0.01 --sfrom -1 --versn $VERSN --model $MODEL --ngpus $GPUS
 $PYTHON_BIN n03_train_conf.py --nfold $i --batch $BATCH --lrate 0.001 --sfrom -1 --versn $VERSN --model $MODEL --ngpus $GPUS
 $PYTHON_BIN n03_train_conf.py --nfold $i --batch $BATCH --lrate 0.0001 --sfrom -1 --versn $VERSN --model $MODEL --ngpus $GPUS

done