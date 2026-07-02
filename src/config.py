import logging
import os
import importlib.util
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_PATH = Path(__file__).with_name("config_local.py")
_USE_MODULE_LOCAL_CONFIG = object()


def _load_local_config(path: Path = LOCAL_CONFIG_PATH):
    """Load optional src/config_local.py without mutating sys.path."""
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location("nmrtrans_config_local", path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

_DEFAULT_ALL_ATOMS = ["B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si"]
_DEFAULT_MAX_SMILES_LENGTH = 80
_DEFAULT_T5_MODEL_NAME = str(PROJECT_ROOT / "models" / "t5-small")


DEFAULT_CONFIG = {
    # Paths
    "MERGED_DATA_DIR": str(PROJECT_ROOT / "cache"),
    "TRAIN_FILE": str(PROJECT_ROOT / "cache" / "train.pkl.lz4"),
    "VAL_FILE": str(PROJECT_ROOT / "cache" / "val.pkl.lz4"),
    "TEST_FILE": str(PROJECT_ROOT / "cache" / "test.pkl.lz4"),
    "VOCAB_PATH": str(PROJECT_ROOT / "vocab.json"),
    "VOCAB_NMRMIND_PATH": str(PROJECT_ROOT / "vocab_nmrmind.json"),
    "SAVE_DIR": str(PROJECT_ROOT / "checkpoints"),

    # Tokenizer and feature selection
    "TOKENIZER_TYPE": "custom",
    "USE_FORMULA_GUIDANCE": True,
    "ALL_ATOMS": _DEFAULT_ALL_ATOMS,
    "FORMULA_VECTOR_SIZE": len(_DEFAULT_ALL_ATOMS),
    "USE_C_NMR": True,
    "USE_H_NMR": True,
    "ENCODER_TYPE": "vanilla_no_pos",

    # Formula encoder
    "FORMULA_ENCODER_D_MODEL": 512,
    "FORMULA_ENCODER_N_LAYERS": 2,
    "FORMULA_ENCODER_N_HEADS": 4,
    "FORMULA_ENCODER_FF_DIM": 1024,
    "FORMULA_ENCODER_DROPOUT": 0.1,

    # Data shape
    "MAX_PEAKS": 60,
    "MAX_SMILES_LENGTH": _DEFAULT_MAX_SMILES_LENGTH,
    "MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS": _DEFAULT_MAX_SMILES_LENGTH + 2,

    # Decoder
    "T5_MODEL_NAME": _DEFAULT_T5_MODEL_NAME,
    "FREEZE_T5_DECODER": False,
    "USE_RANDOM_T5_INIT": False,
    "REMOVE_CROSS_ATTENTION_POSITION_BIAS": True,
    "SHUFFLE_ENCODER_OUTPUTS": False,
    "USE_CUSTOM_DECODER": False,
    "BART_MODEL_NAME": "facebook/bart-base",
    "FREEZE_BART_DECODER": False,
    "USE_RANDOM_BART_INIT": False,
    "BART_D_MODEL": 768,
    "BART_NUM_LAYERS": 6,
    "BART_NUM_HEADS": 12,

    # Peak encoder
    "PEAK_ENCODER_D_MODEL": _infer_t5_d_model(_DEFAULT_T5_MODEL_NAME),
    "PEAK_ENCODER_N_LAYERS": 6,
    "PEAK_ENCODER_N_HEADS": 8,
    "PEAK_ENCODER_FF_DIM": 2048,
    "PEAK_ENCODER_DROPOUT": 0.1,

    # NMR augmentation
    "USE_NMR_JITTER": False,
    "NMR_JITTER_RANGE_C": 2.0,
    "NMR_JITTER_RANGE_H": 0.2,

    # Optimization
    "BATCH_SIZE": 1024,
    "TEST_BATCH_SIZE": 64,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 8000,
    "GRAD_CLIP": 1.0,
    "ACCUM_GRAD_BATCHES": 4,
    "WEIGHT_DECAY": 0.1,
    "NUM_DATA_WORKERS": 8,
    "PREFETCH_FACTOR": 2,
    "LIMIT_VAL_BATCHES": 1.0,
    "CHECK_VAL_EVERY_N_EPOCH": 20,
    "TRAIN_EXAMPLE_LIMIT": 3,
    "TRAIN_EXAMPLE_FREQUENCY": 100,

    # Runtime
    "ACCELERATOR": "gpu",
    "DEVICES": 8,
    "STRATEGY": "ddp_find_unused_parameters_true",
    "PRECISION": "32",
    "NUM_SANITY_VAL_STEPS": 0,
    "LOG_EVERY_N_STEPS": 50,
    "ENABLE_PROGRESS_BAR": True,
    "ENABLE_MODEL_SUMMARY": True,
    "DETERMINISTIC": False,

    # Token IDs populated by prepare_tokenizer()
    "PAD_TOKEN_ID": None,
    "BOS_TOKEN_ID": None,
    "EOS_TOKEN_ID": None,
    "MASK_TOKEN_ID": None,
    "UNK_TOKEN_ID": None,
    "DECODER_START_TOKEN_ID": None,

    # Checkpointing and logging
    "RESUME_CHECKPOINT": None,
    "TEST_CKPT_PATH": None,
    "USE_SWANLAB": False,
    "SWANLAB_PROJECT": "nmrtrans",
    "SWANLAB_RUN_NAME": "t5-small",
    "SWANLAB_INIT_KWARGS": {},
}


_local_config = _load_local_config()


def _copy_default(value):
    return deepcopy(value)


def _local_config_keys(local_config):
    if local_config is None:
        return set()
    return {
        key
        for key in DEFAULT_CONFIG
        if hasattr(local_config, key)
    }


class TrainingConfig:
    """Configuration for NMRTrans training and evaluation."""

    def __init__(self, local_config=_USE_MODULE_LOCAL_CONFIG):
        if local_config is _USE_MODULE_LOCAL_CONFIG:
            local_config = _local_config

        explicit_keys = _local_config_keys(local_config)
        for key, default in DEFAULT_CONFIG.items():
            value = getattr(local_config, key) if key in explicit_keys else _copy_default(default)
            setattr(self, key, value)

        _refresh_derived_values(self, explicit_keys=explicit_keys)


for _config_key, _config_default in DEFAULT_CONFIG.items():
    setattr(TrainingConfig, _config_key, _copy_default(_config_default))


_PATH_KEYS = {
    "MERGED_DATA_DIR",
    "VOCAB_PATH",
    "VOCAB_NMRMIND_PATH",
    "SAVE_DIR",
    "TEST_FILE",
    "RESUME_CHECKPOINT",
    "TEST_CKPT_PATH",
    "T5_MODEL_NAME",
}


def _resolve_config_path(value, base_dir: Path):
    """Resolve relative filesystem paths while leaving simple model names untouched."""
    if value in (None, ""):
        return value

    value_str = str(value)
    if value_str in {"t5-small", "t5-base", "t5-large"}:
        return value

    path = Path(value_str)
    if path.is_absolute():
        return value_str

    return str((base_dir / path).resolve())


def _refresh_derived_values(config: TrainingConfig, explicit_keys=None):
    """Refresh fields that depend on other config values."""
    explicit_keys = set(explicit_keys or [])
    config.TRAIN_FILE = os.path.join(config.MERGED_DATA_DIR, "train.pkl.lz4")
    config.VAL_FILE = os.path.join(config.MERGED_DATA_DIR, "val.pkl.lz4")

    if "TEST_FILE" not in explicit_keys:
        config.TEST_FILE = os.path.join(config.MERGED_DATA_DIR, "test.pkl.lz4")

    config.FORMULA_VECTOR_SIZE = len(config.ALL_ATOMS)
    config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS = config.MAX_SMILES_LENGTH + 2

    if "PEAK_ENCODER_D_MODEL" not in explicit_keys:
        config.PEAK_ENCODER_D_MODEL = _infer_t5_d_model(config.T5_MODEL_NAME)


def apply_config_overrides(config: TrainingConfig, overrides: dict, base_dir=None):
    """Apply YAML overrides to a TrainingConfig instance."""
    if overrides is None:
        return config
    if not isinstance(overrides, dict):
        raise TypeError("Config overrides must be a dictionary")

    base_dir = Path(base_dir or PROJECT_ROOT)
    explicit_keys = set()

    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown config key: {key}")

        if key in _PATH_KEYS:
            value = _resolve_config_path(value, base_dir)

        setattr(config, key, value)
        explicit_keys.add(key)

    _refresh_derived_values(config, explicit_keys=explicit_keys)
    return config


def load_training_config(
    config_path=None,
    base_dir=None,
    logger=None,
    local_config=_USE_MODULE_LOCAL_CONFIG,
) -> TrainingConfig:
    """Create TrainingConfig and optionally overlay values from a single YAML file."""
    config = TrainingConfig(local_config=None if config_path is not None else local_config)
    if config_path is None:
        return config

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("hydra-core/omegaconf is required to load YAML config files") from exc

    base_dir = Path(base_dir or PROJECT_ROOT)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    loaded = OmegaConf.load(config_path)
    overrides = OmegaConf.to_container(loaded, resolve=True)
    apply_config_overrides(config, overrides, base_dir=base_dir)

    if logger:
        logger.info(f"Loaded YAML config: {config_path}")

    return config


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
        if config.TOKENIZER_TYPE == "nmrmind":
            from tokenizer_nmrmind import NMRMindTokenizer
            vocab_path = config.VOCAB_NMRMIND_PATH
            if not vocab_path:
                raise ValueError("配置中缺少VOCAB_NMRMIND_PATH参数，无法加载NMRMind tokenizer")
            logger.info(f"从 {vocab_path} 加载 NMRMindTokenizer")
            tokenizer = NMRMindTokenizer(vocab_path)
        else:
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
