#!/bin/sh

# Check if the YAML filename argument is provided
if [ -z "$1" ]; then
  echo "Error: YAML file name is required as an argument."
  echo "Usage: sbatch -J inat_3 --gpus 1 --mincpus 16 FGDCC_v2.sh <YAML_FILE_NAME>"
  exit 1
fi

# Extract the argument
YAML_FILE_NAME="$1"

# Run the Python script with the specified YAML file
python main_dinov3.py \
  --fname configs/"${YAML_FILE_NAME}.yaml" \
  --devices cuda:0 #cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
