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
import re
from collections import defaultdict, Counter

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


def pad_peak_sequences(peak_sequences, max_peaks):
    """将不等长的峰点序列填充到相同长度，并返回mask（1=valid, 0=pad）。"""
    if not hasattr(pad_peak_sequences, "_logged"):
        logger.info("Peak padding uses 0 values with explicit masks (1=valid, 0=pad).")
        pad_peak_sequences._logged = True
    batch_size = len(peak_sequences)
    if batch_size == 0:
        return torch.zeros(0, max_peaks, 1), torch.zeros(0, max_peaks)
    
    max_len = min(max(len(seq) for seq in peak_sequences if len(seq) > 0), max_peaks)
    
    # 创建填充后的张量，形状为 [batch_size, max_peaks, 1]
    padded = torch.zeros(batch_size, max_peaks, 1)
    mask = torch.zeros(batch_size, max_peaks, dtype=torch.long)
    
    for i, peaks in enumerate(peak_sequences):
        # 只取前max_peaks个峰点
        num_peaks = min(len(peaks), max_peaks)
        if num_peaks > 0:
            padded[i, :num_peaks] = peaks[:num_peaks]
            mask[i, :num_peaks] = 1
    
    return padded, mask

def parse_chemical_formula(formula: str) -> dict:
    """
    Parse chemical formula string to atom counts dictionary.
    Example: "C20H18BrNO2" -> {'C': 20, 'H': 18, 'Br': 1, 'N': 1, 'O': 2}
    """
    if not formula or formula.strip() == "":
        return {}
    
    # 移除空格并处理常见情况
    formula = formula.strip()
    
    # 使用正则表达式匹配原子符号和数量
    # 原子符号总是以大写字母开头，可能跟一个小写字母（如Br, Cl等）
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    atom_counts = defaultdict(int)
    for atom, count in matches:
        count = int(count) if count else 1
        atom_counts[atom] += count
    
    return dict(atom_counts)

def parse_chemical_formula_to_vector(formula: str, atom_mapping: dict) -> torch.Tensor:
    """
    将化学式转换为原子计数向量
    
    Args:
        formula: 化学式字符串，如"C20H18BrNO2"
        atom_mapping: 原子到索引的映射字典
    
    Returns:
        torch.Tensor: 原子计数向量，形状为(formula_vector_size,)
    """
    if not formula or formula.strip() == "":
        return torch.zeros(len(atom_mapping), dtype=torch.float)
    
    # 解析分子式
    atoms = parse_chemical_formula(formula)
    
    # 创建原子计数向量
    vec = torch.zeros(len(atom_mapping), dtype=torch.float)
    for atom, count in atoms.items():
        if atom in atom_mapping:
            idx = atom_mapping[atom]
            vec[idx] = float(count)
    
    return vec

def peaks_collate_fn(batch, tokenizer, config, atom_mapping=None, apply_jitter=False):
    """处理NMR峰值数据集的collate函数，包含化学式向量"""
    # 过滤None值
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # 1. 处理SMILES
    original_smiles_list = [item["original_smiles"] for item in batch]
    tokenized_smiles = []
    for smiles in original_smiles_list:
        tokens = tokenizer.encode(
            smiles, 
            max_length=config.MAX_SMILES_LENGTH, 
            add_special_tokens=True
        )
        tokenized_smiles.append(tokens)
    
    max_len = config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
    padded_smiles = []
    for tokens in tokenized_smiles:
        padded = tokens + [tokenizer.vocab["<pad>"]] * (max_len - len(tokens))
        padded_smiles.append(padded)
    
    smiles_tensor = torch.tensor(padded_smiles, dtype=torch.long)
    
    # 2. 处理谱图数据
    spectra_data = {}
    
    # H-NMR处理
    if "h_nmr_peaks" in batch[0] and batch[0]["h_nmr_peaks"] is not None:
        h_peaks_list = []
        for item in batch:
            if item["h_nmr_peaks"] is not None and len(item["h_nmr_peaks"]) > 0:
                h_peaks = torch.tensor(item["h_nmr_peaks"], dtype=torch.float)
                if apply_jitter and config.NMR_JITTER_RANGE_H > 0:
                    jitter = torch.empty_like(h_peaks).uniform_(-config.NMR_JITTER_RANGE_H, config.NMR_JITTER_RANGE_H)
                    h_peaks = h_peaks + jitter
                h_peaks = h_peaks.unsqueeze(-1)
                h_peaks_list.append(h_peaks)
            else:
                h_peaks_list.append(torch.zeros((0, 1)))
        h_peaks_padded, h_mask = pad_peak_sequences(h_peaks_list, config.MAX_PEAKS)
        spectra_data["h_nmr_peaks"] = h_peaks_padded
        spectra_data["h_nmr_mask"] = h_mask

        if not hasattr(peaks_collate_fn, "_logged_h_sample"):
            sample_peaks = h_peaks_padded[0].squeeze(-1).tolist()
            sample_mask = h_mask[0].tolist()
            logger.info("H-NMR padding sample (first item):")
            logger.info(f"  padded_peaks={sample_peaks}")
            logger.info(f"  mask={sample_mask}")
            peaks_collate_fn._logged_h_sample = True
    
    # C-NMR处理
    if "c_nmr_peaks" in batch[0] and batch[0]["c_nmr_peaks"] is not None:
        c_peaks_list = []
        for item in batch:
            if item["c_nmr_peaks"] is not None and len(item["c_nmr_peaks"]) > 0:
                c_peaks = torch.tensor(item["c_nmr_peaks"], dtype=torch.float)
                if apply_jitter and config.NMR_JITTER_RANGE_C > 0:
                    jitter = torch.empty_like(c_peaks).uniform_(-config.NMR_JITTER_RANGE_C, config.NMR_JITTER_RANGE_C)
                    c_peaks = c_peaks + jitter
                c_peaks = c_peaks.unsqueeze(-1)
                c_peaks_list.append(c_peaks)
            else:
                c_peaks_list.append(torch.zeros((0, 1)))
        c_peaks_padded, c_mask = pad_peak_sequences(c_peaks_list, config.MAX_PEAKS)
        spectra_data["c_nmr_peaks"] = c_peaks_padded
        spectra_data["c_nmr_mask"] = c_mask

        if not hasattr(peaks_collate_fn, "_logged_c_sample"):
            sample_peaks = c_peaks_padded[0].squeeze(-1).tolist()
            sample_mask = c_mask[0].tolist()
            logger.info("C-NMR padding sample (first item):")
            logger.info(f"  padded_peaks={sample_peaks}")
            logger.info(f"  mask={sample_mask}")
            peaks_collate_fn._logged_c_sample = True
    
    # ===== 新增：处理化学式向量 =====
    if config.USE_FORMULA_GUIDANCE and atom_mapping is not None:
        formula_vectors = []
        formula_strings = []  # 用于日志记录
        
        for item in batch:
            formula = item.get("molecular_formula", "")
            formula_strings.append(formula)
            
            # 转换为向量
            vec = parse_chemical_formula_to_vector(formula, atom_mapping)
            formula_vectors.append(vec)
        
        # 堆叠为batch tensor
        formula_tensor = torch.stack(formula_vectors)  # (B, formula_vector_size)
        spectra_data["formula_vector"] = formula_tensor
        spectra_data["formula_strings"] = formula_strings  # 用于验证时显示
    
    return {
        "smiles": smiles_tensor,
        "original_smiles": original_smiles_list,
        **spectra_data
    }


def nmrmind_collate_fn(batch, tokenizer, config, atom_mapping=None, apply_jitter=False):
    """
    Collate function for NMRMind tokenizer (Tokenized Spectra).
    Converts peaks to tokens and creates input_ids directly.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # 1. SMILES Tokenization (Target)
    original_smiles_list = [item["original_smiles"] for item in batch]
    tokenized_smiles = []
    for smiles in original_smiles_list:
        tokens = tokenizer.encode_smiles(
            smiles, 
            add_special_tokens=True
        )
        if len(tokens) > config.MAX_SMILES_LENGTH:
             tokens = tokens[:config.MAX_SMILES_LENGTH]
        tokenized_smiles.append(tokens)
    
    max_len = max(len(t) for t in tokenized_smiles)
    padded_smiles = []
    for tokens in tokenized_smiles:
        padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
        padded_smiles.append(padded)
    
    smiles_tensor = torch.tensor(padded_smiles, dtype=torch.long)
    
    # 2. Input Tokenization (Spectra -> Tokens)
    input_ids_list = []
    
    for item in batch:
        # Get raw peaks
        h_peaks = item.get("h_nmr_peaks")
        c_peaks = item.get("c_nmr_peaks")
        
        # Apply jitter
        if apply_jitter:
            if h_peaks is not None and config.NMR_JITTER_RANGE_H > 0:
                h_peaks = np.array(h_peaks) # Ensure numpy for jitter
                width = config.NMR_JITTER_RANGE_H
                jitter = np.random.uniform(-width, width, size=h_peaks.shape)
                h_peaks = h_peaks + jitter
                h_peaks = h_peaks.tolist()
            if c_peaks is not None and config.NMR_JITTER_RANGE_C > 0:
                c_peaks = np.array(c_peaks)
                width = config.NMR_JITTER_RANGE_C
                jitter = np.random.uniform(-width, width, size=c_peaks.shape)
                c_peaks = c_peaks + jitter
                c_peaks = c_peaks.tolist()
        
        # Encode Spectra
        input_ids = tokenizer.encode_spectra(c_peaks, h_peaks)
        
        # Append Formula if enabled
        if config.USE_FORMULA_GUIDANCE and "molecular_formula" in item:
            formula = item["molecular_formula"]
            if formula:
                f_start_id = tokenizer.token_to_id.get("<molecular_formula>")
                f_end_id = tokenizer.token_to_id.get("</molecular_formula>")
                
                if f_start_id and f_end_id:
                    f_tokens = tokenizer.tokenize_smiles(formula) 
                    f_ids = tokenizer.convert_tokens_to_ids(f_tokens)
                    input_ids = input_ids + [f_start_id] + f_ids + [f_end_id]
        
        input_ids_list.append(input_ids)
        
    # Pad Inputs
    if not input_ids_list:
        return None
        
    max_input_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []
    
    for ids in input_ids_list:
        pad_len = max_input_len - len(ids)
        if pad_len < 0: pad_len = 0
            
        padded_ids = ids + [tokenizer.pad_token_id] * pad_len
        mask = [1] * len(ids) + [0] * pad_len
        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)
    
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
    
    return {
        "smiles": smiles_tensor, # Ground Truth
        "original_smiles": original_smiles_list,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor
    }


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
    
    # 选择 collate_fn
    target_collate_fn = peaks_collate_fn
    if hasattr(config, "TOKENIZER_TYPE") and config.TOKENIZER_TYPE == "nmrmind":
        target_collate_fn = nmrmind_collate_fn
        logger.info("Using nmrmind_collate_fn for tokenized spectra input.")
    else:
        logger.info("Using peaks_collate_fn for continuous spectra input.")

    # 创建带tokenizer和atom_mapping的partial collate函数
    train_collate_fn = partial(
        target_collate_fn,
        tokenizer=tokenizer,
        config=config,
        atom_mapping=atom_mapping,
        apply_jitter=getattr(config, "USE_NMR_JITTER", False),
    )
    val_collate_fn = partial(
        target_collate_fn,
        tokenizer=tokenizer,
        config=config,
        atom_mapping=atom_mapping,
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
            logger.error(f"SwanLabLogger 初始化失败，s将回退到默认日志器: {exc}")
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
