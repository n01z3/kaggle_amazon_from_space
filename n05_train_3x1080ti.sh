#!/usr/bin/env bash

PYTHON_BIN=$HOME/anaconda3/bin/python
BATCH=63
CHECK=220
MODEL=resnext-101-64x4d
VERSN=rnx102_1
GPUS=0,1,2

for i in {7..9}
do
 $PYTHON_BIN n05_train_torchiter.py --nfold $i --batch $BATCH --lrate 0.01 --sfrom i --model $MODEL --versn $VERSN --ngpus $GPUS --trnfc True --nepoh 2
 $PYTHON_BIN n05_train_torchiter.py --nfold $i --batch $BATCH --lrate 0.01 --sfrom -1 --model $MODEL --versn $VERSN --ngpus $GPUS
 $PYTHON_BIN n05_train_torchiter.py --nfold $i --batch $BATCH --lrate 0.001 --sfrom -1 --model $MODEL --versn $VERSN --ngpus $GPUS
 $PYTHON_BIN n05_train_torchiter.py --nfold $i --batch $BATCH --lrate 0.0001 --sfrom -1 --model $MODEL --versn $VERSN --ngpus $GPUS
done
