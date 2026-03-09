#!/bin/bash

# Quick start script for Spectra2Smiles testing
# Simple version without extensive checks

# Activate conda environment and run testing
echo "Activating Conda environment..."
source /mnt/shared-storage-user/yangzhuo/miniconda3/etc/profile.d/conda.sh
conda activate spec2smi

echo "Cleaning up GPU memory before testing..."
pkill -9 -f test.py 2>/dev/null
sleep 2

echo "Checking CUDA version..."
nvcc -V

# Change to project directory
cd "$(dirname "$0")/.."

# Run the testing
python src/test.py --ckpt_path checkpoints/ar-epoch=1661-valacc=val_seq_acc=0.5000.ckpt
