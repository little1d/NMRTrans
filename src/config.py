import logging
import os


def _load_local_config():
    """Load local configuration from config_local.py if it exists."""
    try:
        # Try relative import first
        from . import config_local
        return config_local
    except (ImportError, ValueError):
        # If relative import fails, try absolute import
        try:
            import config_local
            return config_local
        except ImportError:
            # If that also fails, try importing from current directory
            try:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                import config_local
                return config_local
            except ImportError:
                return None


# Load local config if available
_local_config = _load_local_config()

# Debug: check if local config is loaded
if _local_config:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Successfully loaded config_local.py")
    if hasattr(_local_config, 'T5_MODEL_NAME'):
        logger.info(f"T5_MODEL_NAME from config_local: {_local_config.T5_MODEL_NAME}")
else:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("config_local.py not found, using default configuration")


# Helper function to get config value from local config or default
def _get_config(key: str, default):
    """Get configuration value from local config if available, otherwise use default."""
    if _local_config and hasattr(_local_config, key):
        return getattr(_local_config, key)
    return default


def _infer_t5_d_model(model_name: str, fallback: int = 512) -> int:
    """Infer T5 d_model size from model name or path."""
    name = os.path.basename(str(model_name)).lower()
    t5_d_model_map = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
    }
    for key, value in t5_d_model_map.items():
        if key in name:
            return value
    return fallback


# Default paths (will be overridden by config_local.py if it exists)
_DEFAULT_MERGED_DATA_DIR = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/cache/MSD_data"
_DEFAULT_VOCAB_PATH = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/vocab.json"
_DEFAULT_SAVE_DIR = "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles/checkpoints_ar"

# Default SwanLab configuration
_DEFAULT_USE_SWANLAB = True
_DEFAULT_SWANLAB_PROJECT = "spectra2smiles-ar"
_DEFAULT_SWANLAB_RUN_NAME = "t5-ar-baseline"


class TrainingConfig:
    """Configuration for Spectra2Smiles-AR training with T5.
    """
    
    # ========== Path Configuration ==========
    MERGED_DATA_DIR = _get_config("MERGED_DATA_DIR", _DEFAULT_MERGED_DATA_DIR)
    TRAIN_FILE = os.path.join(MERGED_DATA_DIR, "train.pkl.lz4")
    VAL_FILE = os.path.join(MERGED_DATA_DIR, "val.pkl.lz4")
    TEST_FILE = _get_config("TEST_FILE", os.path.join(MERGED_DATA_DIR, "test.pkl.lz4"))
    
    VOCAB_PATH = _get_config("VOCAB_PATH", _DEFAULT_VOCAB_PATH)
    SAVE_DIR = _get_config("SAVE_DIR", _DEFAULT_SAVE_DIR)
    
    # ========== Data Configuration ==========
    
    # ===== 新增：分子式指导配置 =====
    USE_FORMULA_GUIDANCE = _get_config("USE_FORMULA_GUIDANCE", True)
    ALL_ATOMS = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si']
    FORMULA_VECTOR_SIZE = len(ALL_ATOMS)  # 12
    
    # ===== 消融实验配置：控制使用的NMR模态 =====
    USE_C_NMR = _get_config("USE_C_NMR", True)  # 是否使用 C-NMR
    USE_H_NMR = _get_config("USE_H_NMR", True)  # 是否使用 H-NMR
    
    # Formula encoder 配置
    FORMULA_ENCODER_D_MODEL = 512  # 与 peak encoder 相同
    FORMULA_ENCODER_N_LAYERS = 2
    FORMULA_ENCODER_N_HEADS = 4
    FORMULA_ENCODER_FF_DIM = 1024
    FORMULA_ENCODER_DROPOUT = 0.1

    # AR project only uses NMR peaks (discrete)
    MAX_PEAKS = _get_config("MAX_PEAKS", 60)  # Maximum number of peaks per spectrum
    MAX_SMILES_LENGTH = 80
    MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS = MAX_SMILES_LENGTH + 2  # +2 for <bos> and <eos>
    
    # ========== T5 Model Configuration ==========
    T5_MODEL_NAME = _get_config("T5_MODEL_NAME", "t5-small")  # t5-small, t5-base, t5-large
    FREEZE_T5_DECODER = _get_config("FREEZE_T5_DECODER", False)
    USE_RANDOM_T5_INIT = _get_config("USE_RANDOM_T5_INIT", False)
    
    # ========== Peak Encoder Configuration ==========
    PEAK_ENCODER_D_MODEL = _get_config(
        "PEAK_ENCODER_D_MODEL",
        _infer_t5_d_model(_get_config("T5_MODEL_NAME", "t5-small")),
    )
    PEAK_ENCODER_N_LAYERS = _get_config("PEAK_ENCODER_N_LAYERS", 6)  # 增加到6层
    PEAK_ENCODER_N_HEADS = _get_config("PEAK_ENCODER_N_HEADS", 8)  # 增加注意力头
    PEAK_ENCODER_FF_DIM = _get_config("PEAK_ENCODER_FF_DIM", 2048)  # 增加FFN维度
    PEAK_ENCODER_DROPOUT = _get_config("PEAK_ENCODER_DROPOUT", 0.1)  # 添加 dropout
    # ========== NMR Augmentation ==========
    USE_NMR_JITTER = _get_config("USE_NMR_JITTER", False)
    NMR_JITTER_RANGE_C = _get_config("NMR_JITTER_RANGE_C", 2.0)
    NMR_JITTER_RANGE_H = _get_config("NMR_JITTER_RANGE_H", 0.2)
    
    # ========== Training Hyperparameters ==========
    BATCH_SIZE = _get_config("BATCH_SIZE", 1024)
    TEST_BATCH_SIZE = _get_config("TEST_BATCH_SIZE", 64)
    LEARNING_RATE = _get_config("LEARNING_RATE", 1e-4)
    EPOCHS = _get_config("EPOCHS", 6000)
    GRAD_CLIP = _get_config("GRAD_CLIP", 1.0)
    ACCUM_GRAD_BATCHES = _get_config("ACCUM_GRAD_BATCHES", 4)
    
    # Data loading
    NUM_DATA_WORKERS = _get_config("NUM_DATA_WORKERS", 8)
    PREFETCH_FACTOR = _get_config("PREFETCH_FACTOR", 2)
    
    # Validation
    LIMIT_VAL_BATCHES = _get_config("LIMIT_VAL_BATCHES", 1.0)
    CHECK_VAL_EVERY_N_EPOCH = _get_config("CHECK_VAL_EVERY_N_EPOCH", 20)
    
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
        # Import from same directory (src/)
        import sys
        # 获取 src 目录路径（config.py 所在的目录）
        src_path = os.path.dirname(os.path.abspath(__file__))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # 直接从当前目录导入（因为 tokenizer.py 和 config.py 在同一目录）
        from tokenizer import RegexSMILESTokenizer
        
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
