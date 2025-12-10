#!/bin/bash

# Quick start script for Spectra2Smiles training
# Simple version without extensive checks

# Activate conda environment and run training
echo "Activating Conda environment..."
source /mnt/shared-storage-user/yangzhuo/miniconda3/etc/profile.d/conda.sh
conda activate spec2smi

swanlab login --host http://100.101.31.125:8001 --relogin -k PGXG66CPWHASFqnS6irMr

echo "Checking CUDA version..."
nvcc -V

# Change to project directory
cd "$(dirname "$0")/.."

# Run the training
python src/train.py
