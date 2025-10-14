#!/bin/sh

MODEL_TYPE="$1"
EPOCH="$2"

python run_inference_v2.py $MODEL_TYPE $EPOCH