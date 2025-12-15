# Local configuration for Spectra2Smiles-AR
# This file is not tracked by Git

# Data paths (remove _DEFAULT_ prefix!)
MERGED_DATA_DIR = "/mnt/shared-storage-user/yangliujia/Spectra2Smiles-AR/datasets/opendatalab_exp_peaks_no_metal"
VOCAB_PATH = "/mnt/shared-storage-user/yangliujia/spectra_molecule_gen/peaks_to_structure/vocab_regex/vocab.json"
SAVE_DIR = "/mnt/shared-storage-user/yangliujia/Spectra2Smiles-AR/checkpoints"

# SwanLab configuration
USE_SWANLAB = True
SWANLAB_PROJECT = "spectra2smiles"
SWANLAB_RUN_NAME = "12.15-formula-t5-small-ar-baseline"

# Device configuration
DEVICES = 8

# T5 Model - use LOCAL path
T5_MODEL_NAME = "/mnt/shared-storage-user/yangliujia/models/t5-small"

# Training hyperparameters
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4  # 降低学习率，因为现在模型接收更多信息
ACCUM_GRAD_BATCHES = 4  # 梯度累积
GRAD_CLIP = 1.0  # 梯度裁剪

# Peak Encoder hyperparameters (override defaults)
PEAK_ENCODER_N_LAYERS = 6  # 增加到6层（原来2层太浅）
PEAK_ENCODER_N_HEADS = 8   # 增加注意力头
PEAK_ENCODER_FF_DIM = 2048  # 增加FFN维度
PEAK_ENCODER_DROPOUT = 0.1  # 添加dropout防止过拟合

# Validation configuration
CHECK_VAL_EVERY_N_EPOCH = 20