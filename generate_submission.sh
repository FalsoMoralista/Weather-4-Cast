#!/bin/sh

# Run the Python script with the specified YAML file
python generate_submissions_2.py \
  --devices cuda:0 #cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
