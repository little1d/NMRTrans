# Local configuration for Spectra2Smiles-AR
# This file is not tracked by Git

# Data paths (remove _DEFAULT_ prefix!)
MERGED_DATA_DIR = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles-AR/Spectra2Smiles-AR/cache"
VOCAB_PATH = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/vocab.json"
SAVE_DIR = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles-AR/Spectra2Smiles-AR/checkpoints"

# SwanLab configuration
USE_SWANLAB = True
SWANLAB_PROJECT = "spectra2smiles"
SWANLAB_RUN_NAME = "12.10-t5-small-ar-baseline"

# Device configuration
DEVICES = 4

# T5 Model - use LOCAL path
T5_MODEL_NAME = "/mnt/shared-storage-user/yangzhuo/main/models/google-t5/t5-small"
