#!/bin/sh

MODEL=$1
EPOCH=$2
N_BINS=$3
MAX_RAINFALL=$4
STEP=$5

python infer_and_submission.py --model $MODEL --epoch $EPOCH --n_bins $N_BINS --max_rainfall $MAX_RAINFALL --step $STEP

