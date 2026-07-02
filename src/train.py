#!/usr/bin/env python3
"""Entry point for training the Spectra2Smiles-AR model with T5 using new data format."""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from functools import partial

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 完全离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"        # 禁用 Hugging Face Hub
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用进度条

warnings.filterwarnings("ignore")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("RDKit").setLevel(logging.ERROR)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 tokenizer 警告

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from config import TrainingConfig, load_training_config, prepare_tokenizer
from callbacks import get_default_callbacks
from data import MergedDataset
from features import enabled_features_from_config, peaks_collate_fn
from model import NMR2SMILESModel
from runtime import build_trainer_kwargs, validate_runtime_config


# Environment setup
os.environ["TF_DISABLE_MMAP"] = "1"
os.environ["TF_DISABLE_CUBLAS_TENSOR_OP_MATH"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Use local models only
os.environ["HF_DATASETS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')

# Create log directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate timestamp for log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/spectra2smiles_ar_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Rank %(process)d] - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")


def build_dataloaders(config: TrainingConfig, tokenizer):
    """Build data loaders for training and validation with formula guidance."""
    logger.info("Loading datasets...")
    train_dataset = MergedDataset(config.TRAIN_FILE)
    val_dataset = MergedDataset(config.VAL_FILE)

    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")

    # 记录消融实验配置
    use_c_nmr = getattr(config, "USE_C_NMR", True)
    use_h_nmr = getattr(config, "USE_H_NMR", True)
    use_formula = getattr(config, "USE_FORMULA_GUIDANCE", True)
    
    logger.info("\n" + "="*80)
    logger.info("实验配置:")
    logger.info(f"  USE_C_NMR: {use_c_nmr}")
    logger.info(f"  USE_H_NMR: {use_h_nmr}")
    logger.info(f"  USE_FORMULA_GUIDANCE: {use_formula}")
    
    # 验证配置：至少需要启用一个NMR模态（C或H），Formula是可选的
    if not (use_c_nmr or use_h_nmr):
        raise ValueError("至少需要启用一个NMR模态：USE_C_NMR 或 USE_H_NMR（Formula是可选的）")
    logger.info("="*80 + "\n")

    # 创建原子映射
    atom_mapping = None
    if config.USE_FORMULA_GUIDANCE:
        atom_mapping = {atom: idx for idx, atom in enumerate(config.ALL_ATOMS)}
        logger.info(f"✅ 使用分子式指导，原子类型: {config.ALL_ATOMS}")
        logger.info(f"原子映射: {atom_mapping}")
    else:
        logger.info("❌ 未使用分子式指导")
    
    # 使用 peaks_collate_fn 处理连续谱图数据
    logger.info("Using peaks_collate_fn for continuous spectra input.")
    enabled_features = enabled_features_from_config(config)

    # 创建带tokenizer和atom_mapping的partial collate函数
    train_collate_fn = partial(
        peaks_collate_fn,
        tokenizer=tokenizer,
        config=config,
        atom_mapping=atom_mapping,
        enabled_features=enabled_features,
        apply_jitter=getattr(config, "USE_NMR_JITTER", False),
    )
    val_collate_fn = partial(
        peaks_collate_fn,
        tokenizer=tokenizer,
        config=config,
        atom_mapping=atom_mapping,
        enabled_features=enabled_features,
        apply_jitter=False,
    )

    common_kwargs = dict(
        pin_memory=True,
    )
    
    if config.NUM_DATA_WORKERS > 0:
        common_kwargs.update(
            dict(
                num_workers=config.NUM_DATA_WORKERS,
                prefetch_factor=config.PREFETCH_FACTOR,
                persistent_workers=True,
            )
        )
    else:
        common_kwargs["num_workers"] = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate_fn,
        **common_kwargs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_collate_fn,
        **common_kwargs,
    )

    # 验证词汇表范围
    logger.info("\n===== 验证词汇表范围 =====")
    all_ids = []
    for batch in train_loader:
        all_ids.append(batch["smiles"].max().item())
        if len(all_ids) >= 10:  # Just check first 10 batches
            break
    max_id = max(all_ids) if all_ids else 0
    logger.info(f"训练数据中最大token ID (前10批): {max_id}")
    logger.info(f"词汇表大小: {len(tokenizer)}")
    
    if max_id >= len(tokenizer):
        logger.error(f"错误: 最大token ID ({max_id}) 超出词汇表大小 ({len(tokenizer)})")
        logger.error("请检查tokenizer和数据预处理流程")
    
    # 验证formula vector维度
    if config.USE_FORMULA_GUIDANCE:
        sample_batch = next(iter(train_loader))
        if "formula_vector" in sample_batch:
            formula_dim = sample_batch["formula_vector"].shape[1]
            logger.info(f"✅ Formula vector维度验证: {formula_dim} (预期: {config.FORMULA_VECTOR_SIZE})")
            if formula_dim != config.FORMULA_VECTOR_SIZE:
                logger.error(f"维度不匹配! 预期 {config.FORMULA_VECTOR_SIZE}, 实际 {formula_dim}")
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="Train Spectra2Smiles-AR with T5 using new data format")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="可选，Lightning 风格续训/加载的 checkpoint 路径 (.ckpt)",
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        type=str,
        default=None,
        help="可选，单文件 YAML 配置路径；不传时使用 TrainingConfig/config_local.py",
    )
    args = parser.parse_args()

    pl.seed_everything(42)

    config = load_training_config(args.config_path, logger=logger)
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    tokenizer = prepare_tokenizer(config, logger)

    # 传递tokenizer到build_dataloaders
    train_loader, val_loader = build_dataloaders(config, tokenizer)

    # 优先使用命令行 ckpt_path，否则回退到 config.RESUME_CHECKPOINT
    resume_ckpt = args.ckpt_path or getattr(config, "RESUME_CHECKPOINT", None)

    model = NMR2SMILESModel(config, tokenizer)
    
    logger.info(f"\n===== Model Configuration =====")
    logger.info(f"T5 Model: {config.T5_MODEL_NAME}")
    logger.info(f"Peak Encoder d_model: {config.PEAK_ENCODER_D_MODEL}")
    logger.info(f"Peak Encoder layers: {config.PEAK_ENCODER_N_LAYERS}")
    logger.info(f"Peak Encoder heads: {config.PEAK_ENCODER_N_HEADS}")
    logger.info(f"{'=' * 50}\n")

    # Get default callbacks (including checkpoint, printers, and SwanLab logger)
    callbacks = get_default_callbacks(config, config.SAVE_DIR)

    pl_logger = True
    if getattr(config, "USE_SWANLAB", False):
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger
            
            # 检查是否在分布式环境的主进程
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            if local_rank == 0:  # 只在主进程初始化
                init_kwargs = dict(config.SWANLAB_INIT_KWARGS)
                pl_logger = SwanLabLogger(
                    project=config.SWANLAB_PROJECT,
                    experiment_name=config.SWANLAB_RUN_NAME,
                    **init_kwargs,
                )
                logger.info("SwanLabLogger 已启用（主进程）。")
                
                # 立即强制初始化 experiment，避免后续问题
                _ = pl_logger.experiment
                
                # 记录配置到 SwanLab（超参面板）
                try:
                    hparams = {
                        k: getattr(config, k)
                        for k in dir(config)
                        if k.isupper() and not k.startswith("__")
                    }
                    pl_logger.log_hyperparams(hparams)
                except Exception as exc_h:
                    logger.warning(f"超参记录到 SwanLab 失败：{exc_h}")
            else:
                # 在非主进程中，使用 None 或 False
                pl_logger = None
                logger.info(f"SwanLabLogger 未启用（非主进程，local_rank={local_rank}）。")
                
        except ImportError:
            logger.warning(
                "未安装 swanlab，无法启用 SwanLabLogger。请运行 `pip install swanlab` 后重试。"
            )
        except Exception as exc:
            logger.error(f"SwanLabLogger 初始化失败，s将回退到默认日志器: {exc}")
            pl_logger = True

    validate_runtime_config(config)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=pl_logger,
        **build_trainer_kwargs(config),
    )

    num_devices = trainer.num_devices
    if isinstance(num_devices, (list, tuple)):
        num_devices = len(num_devices)

    logger.info(f"\nStarting training on {num_devices} GPUs...")
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(
        f"\nTraining completed in: {int(hours)}h {int(minutes)}m {int(seconds)}s"
    )

    logger.info("\n训练完成！最佳模型已通过 ModelCheckpoint 自动保存。")

if __name__ == "__main__":
    main()
