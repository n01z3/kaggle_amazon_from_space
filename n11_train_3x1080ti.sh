#!/usr/bin/env bash

PYTHON_BIN=$HOME/anaconda3/bin/python
GPUS=0,1,2
MODEL=dpn131
VERSN=dpn131_1
BATCH=84

for i in {0..9}
do
 $PYTHON_BIN n11_train_dpn.py --nfold $i --batch $BATCH --lrate 0.01 --sfrom i --ngpus $GPUS --trnfc True --nepoh 2 --model $MODEL --versn $VERSN
 $PYTHON_BIN n11_train_dpn.py --nfold $i --batch $BATCH --lrate 0.01 --sfrom -1 --ngpus $GPUS --model $MODEL --versn $VERSN
 $PYTHON_BIN n11_train_dpn.py --nfold $i --batch $BATCH --lrate 0.001 --sfrom -1 --ngpus $GPUS --model $MODEL --versn $VERSN
 $PYTHON_BIN n11_train_dpn.py --nfold $i --batch $BATCH --lrate 0.0001 --sfrom -1 --ngpus $GPUS --model $MODEL --versn $VERSN
done