#!/bin/sh

MODEL=$1
EPOCH=$2

python infer_and_submission.py --model $MODEL --epoch $EPOCH
