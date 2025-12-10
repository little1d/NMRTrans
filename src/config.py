import logging
import os
from typing import Dict, List, Union, Tuple, Optional


def _load_local_config():
    """Load local configuration from config_local.py if it exists."""
    try:
        from . import config_local
        return config_local
    except ImportError:
        return None


# Load local config if available
_local_config = _load_local_config()


# Helper function to get config value from local config or default
def _get_config(key: str, default):
    """Get configuration value from local config if available, otherwise use default."""
    if _local_config and hasattr(_local_config, key):
        return getattr(_local_config, key)
    return default


# Default paths (will be overridden by config_local.py if it exists)
_DEFAULT_MERGED_DATA_DIR = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/cache/MSD_data"
_DEFAULT_VOCAB_PATH = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/vocab.json"
_DEFAULT_SAVE_DIR = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/checkpoints_ar"

# Default spectrum and feature configuration
_DEFAULT_SPECTRUM_TYPES = ["h_nmr_peaks", "c_nmr_peaks"]
_DEFAULT_MAX_PEAKS = 60

# Default SwanLab configuration
_DEFAULT_USE_SWANLAB = True
_DEFAULT_SWANLAB_PROJECT = "spectra2smiles-ar"
_DEFAULT_SWANLAB_RUN_NAME = "t5-ar-baseline"


class TrainingConfig:
    """Configuration for Spectra2Smiles-AR training with T5."""
    
    # ========== Path Configuration ==========
    MERGED_DATA_DIR = _get_config("MERGED_DATA_DIR", _DEFAULT_MERGED_DATA_DIR)
    TRAIN_FILE = os.path.join(MERGED_DATA_DIR, "train.pkl.lz4")
    VAL_FILE = os.path.join(MERGED_DATA_DIR, "val.pkl.lz4")
    TEST_FILE = os.path.join(MERGED_DATA_DIR, "test.pkl.lz4")
    
    VOCAB_PATH = _get_config("VOCAB_PATH", _DEFAULT_VOCAB_PATH)
    SAVE_DIR = _get_config("SAVE_DIR", _DEFAULT_SAVE_DIR)
    
    # ========== Data Configuration ==========
    SPECTRUM_TYPES = _get_config("SPECTRUM_TYPES", _DEFAULT_SPECTRUM_TYPES)
    MAX_PEAKS = _get_config("MAX_PEAKS", _DEFAULT_MAX_PEAKS)
    MAX_SMILES_LENGTH = 80
    MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS = MAX_SMILES_LENGTH + 2  # +2 for <bos> and <eos>
    
    # ========== Model Architecture ==========
    # T5 configuration
    T5_MODEL_NAME = _get_config("T5_MODEL_NAME", "t5-small")  # t5-small, t5-base, t5-large
    
    # Peak encoder configuration
    PEAK_ENCODER_D_MODEL = _get_config("PEAK_ENCODER_D_MODEL", 512)
    PEAK_ENCODER_N_LAYERS = _get_config("PEAK_ENCODER_N_LAYERS", 2)
    PEAK_ENCODER_N_HEADS = _get_config("PEAK_ENCODER_N_HEADS", 4)
    PEAK_ENCODER_FF_DIM = _get_config("PEAK_ENCODER_FF_DIM", 1024)
    
    # Whether to freeze T5 decoder initially
    FREEZE_T5_DECODER = _get_config("FREEZE_T5_DECODER", False)
    
    # ========== Training Hyperparameters ==========
    BATCH_SIZE = _get_config("BATCH_SIZE", 32)
    TEST_BATCH_SIZE = _get_config("TEST_BATCH_SIZE", 64)
    LEARNING_RATE = _get_config("LEARNING_RATE", 5e-5)
    EPOCHS = _get_config("EPOCHS", 100)
    GRAD_CLIP = _get_config("GRAD_CLIP", 1.0)
    ACCUM_GRAD_BATCHES = _get_config("ACCUM_GRAD_BATCHES", 1)
    
    # Data loading
    NUM_DATA_WORKERS = _get_config("NUM_DATA_WORKERS", 8)
    PREFETCH_FACTOR = _get_config("PREFETCH_FACTOR", 2)
    
    # Validation
    LIMIT_VAL_BATCHES = _get_config("LIMIT_VAL_BATCHES", 1.0)
    CHECK_VAL_EVERY_N_EPOCH = _get_config("CHECK_VAL_EVERY_N_EPOCH", 1)
    
    # Logging
    TRAIN_EXAMPLE_LIMIT = _get_config("TRAIN_EXAMPLE_LIMIT", 3)
    TRAIN_EXAMPLE_FREQUENCY = _get_config("TRAIN_EXAMPLE_FREQUENCY", 100)
    
    # ========== Device Configuration ==========
    DEVICES = _get_config("DEVICES", 8)  # Number of GPUs
    PRECISION = _get_config("PRECISION", "bf16-mixed")  # "16-mixed", "bf16-mixed", "32"
    
    # ========== Special Token IDs ==========
    # These will be set by prepare_tokenizer()
    PAD_TOKEN_ID = None
    BOS_TOKEN_ID = None
    EOS_TOKEN_ID = None
    MASK_TOKEN_ID = None
    UNK_TOKEN_ID = None
    
    # ========== Checkpoint Configuration ==========
    RESUME_CHECKPOINT = _get_config("RESUME_CHECKPOINT", None)
    TEST_CKPT_PATH = _get_config("TEST_CKPT_PATH", None)
    
    # ========== SwanLab Configuration ==========
    USE_SWANLAB = _get_config("USE_SWANLAB", _DEFAULT_USE_SWANLAB)
    SWANLAB_PROJECT = _get_config("SWANLAB_PROJECT", _DEFAULT_SWANLAB_PROJECT)
    SWANLAB_RUN_NAME = _get_config("SWANLAB_RUN_NAME", _DEFAULT_SWANLAB_RUN_NAME)
    SWANLAB_INIT_KWARGS = _get_config("SWANLAB_INIT_KWARGS", {})


def prepare_tokenizer(config: TrainingConfig, logger: logging.Logger):
    """Load custom RegexSMILESTokenizer and populate config token ids."""
    try:
        # Import from parent package
        import sys
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        
        from spectra2smiles.models.tokenizer import RegexSMILESTokenizer
        
        vocab_path = config.VOCAB_PATH
        if not vocab_path:
            raise ValueError("配置中缺少VOCAB_PATH参数，无法加载自定义tokenizer")
            
        logger.info(f"从 {vocab_path} 加载自定义RegexSMILESTokenizer")
        tokenizer = RegexSMILESTokenizer.from_file(vocab_path)
        
        # Set special token IDs to config
        config.PAD_TOKEN_ID = tokenizer.pad_token_id
        config.BOS_TOKEN_ID = tokenizer.bos_token_id
        config.EOS_TOKEN_ID = tokenizer.eos_token_id
        config.MASK_TOKEN_ID = tokenizer.mask_token_id
        config.UNK_TOKEN_ID = tokenizer.unk_token_id

    except Exception as exc:
        logger.error(f"加载自定义RegexSMILESTokenizer失败: {exc}")
        raise RuntimeError(
            "无法加载自定义tokenizer，请确保词汇表文件存在且格式正确"
        ) from exc

    logger.info("\n===== 自定义RegexSMILESTokenizer信息 =====")
    logger.info(f"词汇表大小: {len(tokenizer)}")
    logger.info(f"PAD token ID: {config.PAD_TOKEN_ID} (token: {tokenizer.pad_token})")
    logger.info(f"BOS token ID: {config.BOS_TOKEN_ID} (token: {tokenizer.bos_token})")
    logger.info(f"EOS token ID: {config.EOS_TOKEN_ID} (token: {tokenizer.eos_token})")
    logger.info(f"MASK token ID: {config.MASK_TOKEN_ID} (token: {tokenizer.mask_token})")
    logger.info(f"UNK token ID: {config.UNK_TOKEN_ID} (token: {tokenizer.unk_token})")
    
    return tokenizer
