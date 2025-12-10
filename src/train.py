#!/usr/bin/env python3
"""Entry point for training the Spectra2Smiles-AR model with T5."""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from config import TrainingConfig, prepare_tokenizer
from callbacks import get_default_callbacks
from data import MergedDataset
from model import NMR2SMILESModel


# Environment setup
os.environ["TF_DISABLE_MMAP"] = "1"
os.environ["TF_DISABLE_CUBLAS_TENSOR_OP_MATH"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Use local models only
os.environ["HF_DATASETS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")

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


def pad_peak_sequences(peak_sequences, max_peaks):
    """将不等长的峰点序列填充到相同长度"""
    batch_size = len(peak_sequences)
    if batch_size == 0:
        return torch.zeros(0, max_peaks, 1)
    
    max_len = min(max(len(seq) for seq in peak_sequences if len(seq) > 0), max_peaks)
    
    # 创建填充后的张量，形状为 [batch_size, max_peaks, 1]
    padded = torch.zeros(batch_size, max_peaks, 1)
    
    for i, peaks in enumerate(peak_sequences):
        # 只取前max_peaks个峰点
        num_peaks = min(len(peaks), max_peaks)
        if num_peaks > 0:
            padded[i, :num_peaks] = peaks[:num_peaks]
    
    return padded


def peaks_collate_fn(batch, tokenizer, config):
    """处理NMR峰值数据集的collate函数"""
    # 过滤None值
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # 1. 使用tokenizer对原始SMILES进行编码
    original_smiles_list = [item["original_smiles"] for item in batch]
    tokenized_smiles = []
    for smiles in original_smiles_list:
        tokens = tokenizer.encode(
            smiles, 
            max_length=config.MAX_SMILES_LENGTH, 
            add_special_tokens=True
        )
        tokenized_smiles.append(tokens)
    
    # 填充到相同长度
    max_len = config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
    padded_smiles = []
    for tokens in tokenized_smiles:
        padded = tokens + [tokenizer.vocab["<pad>"]] * (max_len - len(tokens))
        padded_smiles.append(padded)
    
    # 转换为tensor
    smiles_tensor = torch.tensor(padded_smiles, dtype=torch.long)
    
    # 2. 处理谱图数据
    spectra_data = {}
    
    # 检查数据集中是否包含H-NMR
    if "h_nmr_peaks" in batch[0] and batch[0]["h_nmr_peaks"] is not None:
        h_peaks_list = []
        for item in batch:
            if item["h_nmr_peaks"] is not None and len(item["h_nmr_peaks"]) > 0:
                h_peaks = torch.tensor(item["h_nmr_peaks"], dtype=torch.float).unsqueeze(-1)
                h_peaks_list.append(h_peaks)
            else:
                # 创建一个空的峰点表示
                h_peaks_list.append(torch.zeros((0, 1)))
        
        # 填充到相同长度
        h_peaks_padded = pad_peak_sequences(h_peaks_list, config.MAX_PEAKS)
        spectra_data["h_nmr_peaks"] = h_peaks_padded
    
    # 检查数据集中是否包含C-NMR
    if "c_nmr_peaks" in batch[0] and batch[0]["c_nmr_peaks"] is not None:
        c_peaks_list = []
        for item in batch:
            if item["c_nmr_peaks"] is not None and len(item["c_nmr_peaks"]) > 0:
                c_peaks = torch.tensor(item["c_nmr_peaks"], dtype=torch.float).unsqueeze(-1)
                c_peaks_list.append(c_peaks)
            else:
                # 创建一个空的峰点表示
                c_peaks_list.append(torch.zeros((0, 1)))
        
        # 填充到相同长度
        c_peaks_padded = pad_peak_sequences(c_peaks_list, config.MAX_PEAKS)
        spectra_data["c_nmr_peaks"] = c_peaks_padded
    
    return {
        "smiles": smiles_tensor,
        "original_smiles": original_smiles_list,
        **spectra_data
    }


def build_dataloaders(config: TrainingConfig, tokenizer):
    """Build data loaders for training and validation."""
    logger.info("Loading datasets...")
    train_dataset = MergedDataset(config.TRAIN_FILE)
    val_dataset = MergedDataset(config.VAL_FILE)

    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")

    # 创建带tokenizer的partial collate函数
    collate_fn = partial(peaks_collate_fn, tokenizer=tokenizer, config=config)

    common_kwargs = dict(
        pin_memory=True,
        collate_fn=collate_fn
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
        **common_kwargs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
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
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Spectra2Smiles-AR with T5")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="可选，Lightning 风格续训/加载的 checkpoint 路径 (.ckpt)",
    )
    args = parser.parse_args()

    pl.seed_everything(42)

    config = TrainingConfig()
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
            logger.error(f"SwanLabLogger 初始化失败，将回退到默认日志器: {exc}")
            pl_logger = True

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        callbacks=callbacks,
        logger=pl_logger,
        accelerator="gpu",
        devices=getattr(config, "DEVICES", 4),
        strategy="ddp_find_unused_parameters_true",
        precision=config.PRECISION,
        gradient_clip_val=config.GRAD_CLIP,
        accumulate_grad_batches=config.ACCUM_GRAD_BATCHES,
        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH,
        limit_val_batches=config.LIMIT_VAL_BATCHES,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Set to False for T5 compatibility
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
