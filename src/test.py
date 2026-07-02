#!/usr/bin/env python3
"""Test script for Spectra2Smiles-AR model evaluation."""

import argparse
import logging
import os
import sys
import time
import warnings
import json
import re
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add parent directory to path for imports
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

# Add src directory to path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import TrainingConfig, load_training_config, prepare_tokenizer
from data import MergedDataset
from features import peaks_collate_fn
from model import NMR2SMILESModel

# Environment setup
os.environ["TF_DISABLE_MMAP"] = "1"
os.environ["TF_DISABLE_CUBLAS_TENSOR_OP_MATH"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("RDKit").setLevel(logging.ERROR)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 tokenizer 警告

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_ar_inference.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# RDKit for SMILES validation and similarity
try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import RDLogger
    
    # Disable RDKit error/warning messages to stderr
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.debug')
    
    RDKit_AVAILABLE = True
    MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    logger.info("RDKit is available. SMILES validity and similarity will be checked.")
except ImportError:
    logger.warning("RDKit is not installed. Cannot validate SMILES or calculate similarity. Please run `pip install rdkit` to enable this feature.")
    RDKit_AVAILABLE = False
    MORGAN_GENERATOR = None
except AttributeError:
    # Older versions of RDKit might not have RDLogger
    RDKit_AVAILABLE = True
    MORGAN_GENERATOR = None
    logger.info("RDKit is available. SMILES validity and similarity will be checked.")

def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is chemically valid, automatically remove special tokens like <bos>, <eos>, <pad>"""
    if not RDKit_AVAILABLE or not smiles:
        return False
    
    # Clean special tokens: remove <bos>, <eos>, <pad>, <mask> etc.
    clean_smiles = re.sub(r'<bos>|<eos>|<pad>|<mask>', '', smiles)
    # Remove possible extra spaces and special characters
    clean_smiles = clean_smiles.strip()
    
    # Check if cleaned string is empty
    if not clean_smiles:
        return False
    
    try:
        # Suppress RDKit error messages by catching exceptions silently
        mol = Chem.MolFromSmiles(clean_smiles, sanitize=True)
        return mol is not None
    except (ValueError, RuntimeError, AttributeError, TypeError):
        # Silently handle parsing errors
        return False
    except Exception:
        # Catch any other exceptions
        return False

def calculate_smiles_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two SMILES strings using RDKit.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
    
    Returns:
        Similarity score between 0.0 and 1.0, or -1.0 if either SMILES is invalid
    """
    if not RDKit_AVAILABLE:
        return -1.0
    
    # Clean SMILES strings
    clean_smiles1 = re.sub(r'<bos>|<eos>|<pad>|<mask>', '', smiles1).strip()
    clean_smiles2 = re.sub(r'<bos>|<eos>|<pad>|<mask>', '', smiles2).strip()
    
    if not clean_smiles1 or not clean_smiles2:
        return -1.0
    
    try:
        # Parse molecules (suppress RDKit error messages)
        mol1 = Chem.MolFromSmiles(clean_smiles1, sanitize=True)
        mol2 = Chem.MolFromSmiles(clean_smiles2, sanitize=True)
        
        if mol1 is None or mol2 is None:
            return -1.0
        
        if MORGAN_GENERATOR is None:
            return -1.0

        fp1 = MORGAN_GENERATOR.GetFingerprint(mol1)
        fp2 = MORGAN_GENERATOR.GetFingerprint(mol2)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return float(similarity)
    
    except (ValueError, RuntimeError, AttributeError, TypeError):
        # Silently handle parsing errors
        return -1.0
    except Exception:
        # Catch any other exceptions
        return -1.0

def shard_indices(total_size: int, num_shards: int, shard_index: int):
    """Return deterministic round-robin indices for one evaluation shard."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    return list(range(shard_index, total_size, num_shards))


def build_test_dataloader(
    config: TrainingConfig,
    tokenizer,
    test_file=None,
    enabled_features=None,
    shard_index: int = 0,
    num_shards: int = 1,
):
    """
    Build test data loader
    
    Args:
        config: TrainingConfig instance
        tokenizer: Tokenizer instance
        test_file: Path to test file (optional)
        enabled_features: Set of enabled features, e.g., {'c_nmr', 'formula'}
                         If None, uses all features based on model configuration
        shard_index: Evaluation shard rank.
        num_shards: Number of evaluation shards.
    """
    # Use provided test_file or fall back to config.TEST_FILE
    test_file_path = test_file or config.TEST_FILE
    logger.info(f"Loading test dataset from: {test_file_path}")
    
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    
    full_dataset = MergedDataset(test_file_path)
    logger.info(f"Test set size: {len(full_dataset)}")
    
    # Check data structure
    if len(full_dataset) > 0:
        sample = full_dataset[0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        has_source = "source" in sample
        has_formula = "molecular_formula" in sample
        has_h_nmr = "h_nmr_peaks" in sample
        has_c_nmr = "c_nmr_peaks" in sample
        logger.info(f"  - Has 'source' field: {has_source}")
        logger.info(f"  - Has 'molecular_formula' field: {has_formula}")
        logger.info(f"  - Has 'h_nmr_peaks' field: {has_h_nmr}")
        logger.info(f"  - Has 'c_nmr_peaks' field: {has_c_nmr}")

    test_dataset = full_dataset
    if num_shards > 1:
        indices = shard_indices(len(full_dataset), num_shards, shard_index)
        test_dataset = Subset(full_dataset, indices)
        logger.info(
            f"Evaluation shard {shard_index + 1}/{num_shards}: "
            f"{len(indices)} samples"
        )
    
    # Create atom mapping for formula
    atom_mapping = None
    if config.USE_FORMULA_GUIDANCE:
        atom_mapping = {atom: idx for idx, atom in enumerate(config.ALL_ATOMS)}
        logger.info(f"Using formula guidance with atoms: {config.ALL_ATOMS}")
    
    # Log enabled features
    if enabled_features:
        logger.info(f"Enabled features: {enabled_features}")
    else:
        logger.info("Using all available features based on model configuration")
    
    # Create collate function
    from functools import partial
    collate_fn = partial(
        peaks_collate_fn,
        tokenizer=tokenizer,
        config=config,
        atom_mapping=atom_mapping,
        enabled_features=enabled_features
    )
    
    # Use num_workers=0 for testing to avoid multiprocessing issues in container environments
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Disable multiprocessing to avoid "Address already in use" errors
        pin_memory=True,
    )
    
    return test_loader

def get_model_module(model):
    """
    Get the underlying model module, handling DataParallel wrapper.
    
    Args:
        model: Model instance, possibly wrapped in DataParallel
    
    Returns:
        The actual model module
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

def load_model(
    config: TrainingConfig,
    tokenizer,
    checkpoint_path: str,
    device: torch.device | str | None = None,
    use_data_parallel: bool = True,
):
    """Load trained model"""
    logger.info(f"\n===== Loading Model =====")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    model = NMR2SMILESModel(config, tokenizer)
    
    if os.path.exists(checkpoint_path):
        # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain config objects
        # Set weights_only=False for trusted checkpoints
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            # For older PyTorch versions that don't support weights_only
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get state_dict from checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Filter state_dict to only include keys that exist in the current model
        # This handles cases where checkpoint was saved with different encoder configurations
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                # Check if shapes match
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    logger.warning(f"Shape mismatch for {key}: checkpoint has {value.shape}, model expects {model_state_dict[key].shape}. Skipping.")
                    unexpected_keys.append(key)
            else:
                # Key not in current model (e.g., c_encoder when USE_C_NMR=False)
                unexpected_keys.append(key)
        
        # Check for missing keys (keys in model but not in checkpoint)
        for key in model_state_dict.keys():
            if key not in filtered_state_dict:
                missing_keys.append(key)
        
        # Log information
        if unexpected_keys:
            logger.info(f"Skipping {len(unexpected_keys)} keys from checkpoint that don't exist in current model:")
            # Group by prefix for cleaner logging
            unexpected_prefixes = {}
            for key in unexpected_keys:
                prefix = key.split('.')[0]
                if prefix not in unexpected_prefixes:
                    unexpected_prefixes[prefix] = []
                unexpected_prefixes[prefix].append(key)
            
            for prefix, keys in unexpected_prefixes.items():
                logger.info(f"  - {prefix}: {len(keys)} keys (e.g., {keys[0]})")
        
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys in checkpoint (will use random initialization):")
            for key in missing_keys[:10]:  # Show first 10
                logger.warning(f"  - {key}")
            if len(missing_keys) > 10:
                logger.warning(f"  ... and {len(missing_keys) - 10} more")
        
        # Load filtered state_dict with strict=False to allow partial loading
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} keys from checkpoint")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is not None:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available")
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded to {device}")
        return model

    # Determine device and setup multi-GPU if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
        model = model.to(device)
        model.eval()
        return model
    
    # Multi-GPU support using DataParallel
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPU(s)")
    
    if use_data_parallel and num_gpus > 1:
        # DataParallel requires model to be on cuda:0, then it distributes to all GPUs
        device = torch.device("cuda:0")
        model = model.to(device)
        model.eval()
        
        # Wrap with DataParallel to use all GPUs
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
        
        # Log which GPUs are being used
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        logger.info(f"Using {num_gpus} GPUs for parallel inference: {', '.join(gpu_list)}")
        logger.info(f"Primary device: {device}")
        logger.info("For true multi-GPU generation over the test set, use --parallel_eval.")
    else:
        device = torch.device("cuda:0")
        model = model.to(device)
        model.eval()
        logger.info(f"Using single GPU: {device}")
    
    logger.info(f"Model loaded to {device}")
    return model

def evaluate_autoregressive_generation(model, test_loader, config, tokenizer, enabled_features=None):
    """
    Evaluate model using autoregressive generation with greedy decoding.
    
    Args:
        model: Trained model instance
        test_loader: DataLoader for test data
        config: TrainingConfig instance
        tokenizer: Tokenizer instance
        enabled_features: Set of enabled features, e.g., {'c_nmr', 'formula'}
                         If None, uses all features based on model configuration
    
    Uses greedy decoding (num_beams=1, do_sample=False) for fast evaluation.
    """
    logger.info("\n===== Starting Autoregressive Generation Evaluation =====")
    
    # enabled_features should already be validated in main(), but double-check for safety
    if enabled_features is None:
        raise ValueError("enabled_features must be specified (should not be None at this point)")
    
    # Ensure at least one NMR feature is enabled (safety check)
    if not ('c_nmr' in enabled_features or 'h_nmr' in enabled_features):
        raise ValueError("At least one NMR feature (c_nmr or h_nmr) must be enabled")
    
    logger.info(f"Using features: {enabled_features}")
    start_time = time.time()
    
    # Get model module and device (handle DataParallel wrapper)
    model_module = get_model_module(model)
    device = next(model_module.parameters()).device
    
    # Log GPU usage information
    if isinstance(model, torch.nn.DataParallel):
        num_gpus = len(model.device_ids)
        gpu_list = [f"cuda:{i}" for i in model.device_ids]
        logger.info(f"Using {num_gpus} GPUs: {', '.join(gpu_list)} (primary: {device})")
    else:
        logger.info(f"Using device: {device}")
    
    # Greedy decoding parameters
    gen_kwargs = {
        "num_beams": 1,
        "do_sample": False,
    }
    # Beam search配置，用于计算top-k序列准确率
    beam_sizes = [3, 5, 10]
    max_beam = max(beam_sizes)
    topk_hits = {k: 0 for k in beam_sizes}
    total_samples = 0
    
    # Initialize metrics
    metrics = {
        "token_acc": [],
        "seq_acc": [],
        "valid_smiles": [],
        "similarity": [],  # Tanimoto similarity with ground truth
        "examples": []
    }
    # Teacher forcing metrics（与训练/validation 相同的评估方式）
    tf_metrics = {
        "token_acc": [],
        "seq_acc": [],
    }
    
    # Grouped metrics by SMILES length
    group_metrics = {
        "all": {
            "token_acc": [],
            "seq_acc": [],
            "valid_smiles": [],
            "similarity": [],
        },
        "short_0-20": {
            "token_acc": [],
            "seq_acc": [],
            "valid_smiles": [],
            "similarity": [],
        },
        "medium_20-40": {
            "token_acc": [],
            "seq_acc": [],
            "valid_smiles": [],
            "similarity": [],
        },
        "long_40-60": {
            "token_acc": [],
            "seq_acc": [],
            "valid_smiles": [],
            "similarity": [],
        },
        "very_long_>60": {
            "token_acc": [],
            "seq_acc": [],
            "valid_smiles": [],
            "similarity": [],
        }
    }
    
    example_count = 0
    max_examples = 100
    
    def _clean_content_tokens(tokens: torch.Tensor, cfg):
        """去掉 PAD/BOS/EOS，只保留有效内容token。"""
        mask = (
            (tokens != cfg.PAD_TOKEN_ID) &
            (tokens != cfg.BOS_TOKEN_ID) &
            (tokens != cfg.EOS_TOKEN_ID)
        )
        return tokens[mask]
    
    # Create progress bar with total number of batches
    total_batches = len(test_loader)
    pbar = tqdm(test_loader, desc="Evaluating", total=total_batches, 
               unit="batch", ncols=100, leave=True, dynamic_ncols=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            smiles_ids = batch["smiles"].long()
            original_smiles_list = batch["original_smiles"]
            
            # Get inputs based on enabled features
            c_peaks = batch.get("c_nmr_peaks") if 'c_nmr' in enabled_features else None
            h_features = batch.get("h_nmr_features") if 'h_nmr' in enabled_features else None
            c_nmr_mask = batch.get("c_nmr_mask") if 'c_nmr' in enabled_features else None
            h_nmr_mask = batch.get("h_nmr_mask") if 'h_nmr' in enabled_features else None
            formula_vector = batch.get("formula_vector") if 'formula' in enabled_features else None
            
            batch_size = smiles_ids.size(0)
            total_samples += batch_size
            
            try:
                # ===== Teacher forcing 评估（与训练/validation 对齐）=====
                labels = smiles_ids.clone()
                labels[labels == config.PAD_TOKEN_ID] = -100
                tf_outputs = model_module(
                    c_peaks=c_peaks,
                    h_features=h_features,
                    formula_vector=formula_vector,
                    smiles_ids=labels,
                    c_nmr_mask=c_nmr_mask,
                    h_nmr_mask=h_nmr_mask,
                )
                tf_logits = tf_outputs.logits
                tf_pred_tokens = tf_logits.argmax(dim=-1)
                tf_valid_mask = (smiles_ids != config.PAD_TOKEN_ID)
                if tf_valid_mask.sum() > 0:
                    tf_correct = (tf_pred_tokens == smiles_ids) & tf_valid_mask
                    tf_token_acc = tf_correct.sum().float() / tf_valid_mask.sum().float()
                else:
                    tf_token_acc = torch.tensor(0.0, device=device)
                tf_seq_correct = ((tf_pred_tokens == smiles_ids) | ~tf_valid_mask).all(dim=1)
                tf_seq_acc = tf_seq_correct.float().mean()
                tf_metrics["token_acc"].append(tf_token_acc.item())
                tf_metrics["seq_acc"].append(tf_seq_acc.item())

                # Generate sequences using greedy decoding
                generated_ids = model_module.generate(
                    c_peaks=c_peaks,
                    h_features=h_features,
                    formula_vector=formula_vector,
                    max_length=config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS,
                    c_nmr_mask=c_nmr_mask,
                    h_nmr_mask=h_nmr_mask,
                    **gen_kwargs
                )
                
                # 额外：Beam search 获取 top-k 序列（用于 top3/5/10 seq acc）
                beam_generated_ids = model_module.generate(
                    c_peaks=c_peaks,
                    h_features=h_features,
                    formula_vector=formula_vector,
                    max_length=config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS,
                    num_beams=max_beam,
                    num_return_sequences=max_beam,
                    do_sample=False,
                    early_stopping=True,
                    c_nmr_mask=c_nmr_mask,
                    h_nmr_mask=h_nmr_mask,
                )
                # 形状调整为 (B, max_beam, seq_len)
                beam_generated_ids = beam_generated_ids.view(batch_size, max_beam, -1)
                
                # Calculate metrics for each sample
                for i in range(batch_size):
                    true_tokens = smiles_ids[i]
                    gen_tokens = generated_ids[i]
                    beam_tokens = beam_generated_ids[i]
                    
                    # Convert to SMILES strings
                    true_smiles = model_module.tokens_to_smiles(true_tokens)
                    gen_smiles = model_module.tokens_to_smiles(gen_tokens)
                    
                    # Calculate true length (excluding special tokens)
                    true_len = ((true_tokens != config.PAD_TOKEN_ID) & 
                               (true_tokens != config.BOS_TOKEN_ID) & 
                               (true_tokens != config.EOS_TOKEN_ID)).sum().item()
                    
                    # Create valid mask for true tokens (exclude PAD, BOS, EOS)
                    valid_mask = (
                        (true_tokens != config.PAD_TOKEN_ID) &
                        (true_tokens != config.BOS_TOKEN_ID) &
                        (true_tokens != config.EOS_TOKEN_ID)
                    )
                    
                    # Token accuracy: compare aligned sequences
                    # Find the actual content length (excluding padding)
                    true_content_len = valid_mask.sum().item()
                    
                    # For generated tokens, also exclude special tokens
                    gen_valid_mask = (
                        (gen_tokens != config.PAD_TOKEN_ID) &
                        (gen_tokens != config.BOS_TOKEN_ID) &
                        (gen_tokens != config.EOS_TOKEN_ID)
                    )
                    gen_content_len = gen_valid_mask.sum().item()
                    
                    # Get content tokens (excluding special tokens)
                    true_content = true_tokens[valid_mask]
                    gen_content = gen_tokens[gen_valid_mask]
                    
                    # Token accuracy: compare aligned sequences
                    min_content_len = min(true_content_len, gen_content_len)
                    
                    if min_content_len > 0:
                        # Compare up to minimum length
                        true_content_compare = true_content[:min_content_len]
                        gen_content_compare = gen_content[:min_content_len]
                        correct = (true_content_compare == gen_content_compare)
                        token_acc = correct.sum().float() / min_content_len
                    else:
                        token_acc = torch.tensor(0.0)
                    
                    # Sequence accuracy: exact match requires same length AND all tokens match
                    if true_content_len == gen_content_len and true_content_len > 0:
                        seq_acc = (true_content == gen_content).all().item()
                    else:
                        seq_acc = 0.0
                    
                    # SMILES validity
                    is_valid = is_valid_smiles(gen_smiles)
                    
                    # Calculate Tanimoto similarity with ground truth
                    similarity = calculate_smiles_similarity(true_smiles, gen_smiles)
                    
                    # ===== Beam search top-k 序列准确率（精确匹配内容token）=====
                    true_content = _clean_content_tokens(true_tokens, config)
                    for k in beam_sizes:
                        candidates = beam_tokens[:k]  # (k, seq_len)
                        hit = False
                        for cand in candidates:
                            cand_content = _clean_content_tokens(cand, config)
                            if cand_content.shape[0] == true_content.shape[0] and torch.equal(cand_content, true_content):
                                hit = True
                                break
                        if hit:
                            topk_hits[k] += 1
                    
                    # Record metrics
                    metrics["token_acc"].append(token_acc.item())
                    metrics["seq_acc"].append(seq_acc)
                    metrics["valid_smiles"].append(1 if is_valid else 0)
                    if similarity >= 0:  # Only record valid similarities
                        metrics["similarity"].append(similarity)
                    
                    # Group by length
                    if true_len < 20:
                        group = "short_0-20"
                    elif true_len < 40:
                        group = "medium_20-40"
                    elif true_len < 60:
                        group = "long_40-60"
                    else:
                        group = "very_long_>60"
                    
                    group_metrics[group]["token_acc"].append(token_acc.item())
                    group_metrics[group]["seq_acc"].append(seq_acc)
                    group_metrics[group]["valid_smiles"].append(1 if is_valid else 0)
                    if similarity >= 0:
                        group_metrics[group]["similarity"].append(similarity)
                    
                    group_metrics["all"]["token_acc"].append(token_acc.item())
                    group_metrics["all"]["seq_acc"].append(seq_acc)
                    group_metrics["all"]["valid_smiles"].append(1 if is_valid else 0)
                    if similarity >= 0:
                        group_metrics["all"]["similarity"].append(similarity)
                    
                    # Save examples (up to max_examples)
                    if example_count < max_examples:
                        metrics["examples"].append({
                            "original": original_smiles_list[i],
                            "generated": gen_smiles,
                            "token_acc": token_acc.item(),
                            "seq_acc": seq_acc,
                            "is_valid": is_valid,
                            "similarity": similarity if similarity >= 0 else None,
                            "true_length": true_len,
                        })
                        example_count += 1
            
            except Exception as e:
                logger.warning(f"Error in generation for batch {batch_idx}: {e}")
                continue
        
        # Close progress bar
        pbar.close()
    
    # Calculate average metrics
    if len(metrics["token_acc"]) == 0:
        raise ValueError("No samples were evaluated. Please check your data loader.")
    
    # Calculate average similarity (only for valid similarities)
    avg_similarity = None
    if len(metrics["similarity"]) > 0:
        avg_similarity = np.mean(metrics["similarity"])
    
    final_results = {
        "token_accuracy": np.mean(metrics["token_acc"]),
        "sequence_accuracy": np.mean(metrics["seq_acc"]),
        "valid_smiles_ratio": np.mean(metrics["valid_smiles"]),
        "tanimoto_similarity": float(avg_similarity) if avg_similarity is not None else None,
        "similarity_samples": len(metrics["similarity"]),
        "total_samples": len(metrics["token_acc"]),
        "examples": metrics["examples"][:max_examples],
        # Teacher forcing metrics
        "teacher_forcing_token_accuracy": np.mean(tf_metrics["token_acc"]) if len(tf_metrics["token_acc"]) > 0 else None,
        "teacher_forcing_sequence_accuracy": np.mean(tf_metrics["seq_acc"]) if len(tf_metrics["seq_acc"]) > 0 else None,
        # Beam search top-k seq accuracy
        "topk_sequence_accuracy": {k: topk_hits[k] / total_samples for k in topk_hits},
    }
    
    # Calculate grouped metrics
    grouped_results = {}
    for group_name, group_data in group_metrics.items():
        if len(group_data["token_acc"]) == 0:
            continue
        
        grouped_results[group_name] = {
            "sample_count": len(group_data["token_acc"]),
            "token_acc": np.mean(group_data["token_acc"]),
            "seq_acc": np.mean(group_data["seq_acc"]),
            "valid_ratio": np.mean(group_data["valid_smiles"]),
        }
        
        if len(group_data["similarity"]) > 0:
            grouped_results[group_name]["similarity"] = np.mean(group_data["similarity"])
    
    inference_time = time.time() - start_time
    
    return {
        "metrics": final_results,
        "grouped_metrics": grouped_results,
        "inference_time": inference_time
    }

def print_results(results, enabled_features=None):
    """Print evaluation results"""
    logger.info("\n" + "="*80)
    logger.info("===== Evaluation Results (Greedy Decoding) =====")
    logger.info("="*80)
    
    if enabled_features:
        logger.info(f"\nEnabled Features: {enabled_features}")
    
    # Print overall results
    metrics = results["metrics"]
    logger.info(f"\n--- Overall Results ---")
    logger.info(f"  Token Accuracy: {metrics['token_accuracy']:.4f}")
    logger.info(f"  Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
    logger.info(f"  Valid SMILES Ratio: {metrics['valid_smiles_ratio']:.4f}")
    if metrics.get('tanimoto_similarity') is not None:
        logger.info(f"  Tanimoto Similarity: {metrics['tanimoto_similarity']:.4f} (n={metrics['similarity_samples']})")
    if metrics.get("teacher_forcing_token_accuracy") is not None:
        logger.info(f"  TF Token Accuracy (teacher forcing): {metrics['teacher_forcing_token_accuracy']:.4f}")
    if metrics.get("teacher_forcing_sequence_accuracy") is not None:
        logger.info(f"  TF Sequence Accuracy (teacher forcing): {metrics['teacher_forcing_sequence_accuracy']:.4f}")
    if metrics.get("topk_sequence_accuracy"):
        topk_items = sorted(
            metrics["topk_sequence_accuracy"].items(),
            key=lambda item: int(item[0]),
        )
        topk_str = ", ".join([f"top{k}: {value:.4f}" for k, value in topk_items])
        logger.info(f"  Beam Search Seq Acc ({topk_str})")
    logger.info(f"  Total Samples: {metrics['total_samples']}")
    
    # Print grouped metrics
    logger.info("\n--- Grouped Metrics by SMILES Length ---")
    for group_name, group_metrics in results["grouped_metrics"].items():
        if group_name == "all":
            continue
        logger.info(f"\n  {group_name} (n={group_metrics['sample_count']}):")
        logger.info(f"    Token Acc: {group_metrics['token_acc']:.4f}")
        logger.info(f"    Seq Acc: {group_metrics['seq_acc']:.4f}")
        logger.info(f"    Valid Ratio: {group_metrics['valid_ratio']:.4f}")
        if 'similarity' in group_metrics:
            logger.info(f"    Tanimoto Similarity: {group_metrics['similarity']:.4f}")
    
    logger.info(f"\nInference Time: {results['inference_time']:.2f} seconds")
    logger.info("="*80)

def parse_features(features_str):
    """
    Parse features string into a set of enabled features.
    
    Args:
        features_str: Comma-separated string, e.g., "c_nmr,formula" or "h_nmr,c_nmr"
    
    Returns:
        Set of enabled feature names
    """
    if features_str is None or features_str.strip() == "":
        return None
    
    valid_features = {'c_nmr', 'h_nmr', 'formula'}
    features = set(f.strip().lower() for f in features_str.split(','))
    
    # Validate features
    invalid_features = features - valid_features
    if invalid_features:
        raise ValueError(
            f"Invalid features: {invalid_features}. "
            f"Valid features are: {valid_features}"
        )
    
    # Ensure at least one NMR feature is specified
    if not ('c_nmr' in features or 'h_nmr' in features):
        raise ValueError(
            "At least one NMR feature (c_nmr or h_nmr) must be specified. "
            f"Specified features: {features}"
        )
    
    return features


def resolve_enabled_features(requested_features, model_module):
    """Resolve final evaluation features from CLI override or model encoders."""
    if requested_features is None:
        enabled_features = set()
        if model_module.c_encoder is not None:
            enabled_features.add("c_nmr")
        if model_module.h_encoder is not None:
            enabled_features.add("h_nmr")
        if model_module.formula_encoder is not None:
            enabled_features.add("formula")
        return enabled_features

    if "c_nmr" in requested_features and model_module.c_encoder is None:
        raise ValueError(
            "Feature 'c_nmr' is requested but model does not have c_encoder. "
            "Please check your checkpoint or use different features."
        )
    if "h_nmr" in requested_features and model_module.h_encoder is None:
        raise ValueError(
            "Feature 'h_nmr' is requested but model does not have h_encoder. "
            "Please check your checkpoint or use different features."
        )
    if "formula" in requested_features and model_module.formula_encoder is None:
        raise ValueError(
            "Feature 'formula' is requested but model does not have formula_encoder. "
            "Please check your checkpoint or use different features."
        )
    if not ("c_nmr" in requested_features or "h_nmr" in requested_features):
        raise ValueError(
            "At least one NMR feature (c_nmr or h_nmr) must be enabled. "
            f"Specified features: {requested_features}"
        )

    return set(requested_features)


def _weighted_average(shard_results, metric_key, count_key="total_samples"):
    weighted_sum = 0.0
    count_sum = 0
    for result in shard_results:
        metrics = result["metrics"]
        value = metrics.get(metric_key)
        count = int(metrics.get(count_key, 0))
        if value is None or count <= 0:
            continue
        weighted_sum += float(value) * count
        count_sum += count
    if count_sum == 0:
        return None
    return weighted_sum / count_sum


def merge_evaluation_results(shard_results):
    """Merge independently evaluated dataset shards into one result payload."""
    if not shard_results:
        raise ValueError("No shard results to merge")

    total_samples = sum(int(result["metrics"].get("total_samples", 0)) for result in shard_results)
    if total_samples <= 0:
        raise ValueError("No samples were evaluated across shards")

    similarity_samples = sum(int(result["metrics"].get("similarity_samples", 0)) for result in shard_results)
    tanimoto_similarity = None
    if similarity_samples > 0:
        tanimoto_similarity = sum(
            float(result["metrics"]["tanimoto_similarity"]) * int(result["metrics"].get("similarity_samples", 0))
            for result in shard_results
            if result["metrics"].get("tanimoto_similarity") is not None
        ) / similarity_samples

    examples = []
    for result in shard_results:
        examples.extend(result["metrics"].get("examples", []))

    topk_keys = set()
    for result in shard_results:
        topk_keys.update(result["metrics"].get("topk_sequence_accuracy", {}).keys())
    topk_sequence_accuracy = {}
    for key in sorted(topk_keys, key=lambda item: int(item)):
        weighted_sum = 0.0
        count_sum = 0
        for result in shard_results:
            metrics = result["metrics"]
            topk = metrics.get("topk_sequence_accuracy", {})
            if key in topk:
                value = topk[key]
            elif str(key) in topk:
                value = topk[str(key)]
            else:
                continue
            count = int(metrics.get("total_samples", 0))
            weighted_sum += float(value) * count
            count_sum += count
        if count_sum > 0:
            topk_sequence_accuracy[int(key)] = weighted_sum / count_sum

    merged_metrics = {
        "token_accuracy": _weighted_average(shard_results, "token_accuracy"),
        "sequence_accuracy": _weighted_average(shard_results, "sequence_accuracy"),
        "valid_smiles_ratio": _weighted_average(shard_results, "valid_smiles_ratio"),
        "tanimoto_similarity": tanimoto_similarity,
        "similarity_samples": similarity_samples,
        "total_samples": total_samples,
        "examples": examples[:100],
        "teacher_forcing_token_accuracy": _weighted_average(
            shard_results,
            "teacher_forcing_token_accuracy",
        ),
        "teacher_forcing_sequence_accuracy": _weighted_average(
            shard_results,
            "teacher_forcing_sequence_accuracy",
        ),
        "topk_sequence_accuracy": topk_sequence_accuracy,
    }

    group_names = []
    for result in shard_results:
        for group_name in result.get("grouped_metrics", {}):
            if group_name not in group_names:
                group_names.append(group_name)

    grouped_metrics = {}
    for group_name in group_names:
        group_count = 0
        accumulators = {
            "token_acc": 0.0,
            "seq_acc": 0.0,
            "valid_ratio": 0.0,
            "similarity": 0.0,
        }
        similarity_count = 0

        for result in shard_results:
            group = result.get("grouped_metrics", {}).get(group_name)
            if not group:
                continue
            count = int(group.get("sample_count", 0))
            if count <= 0:
                continue
            group_count += count
            accumulators["token_acc"] += float(group.get("token_acc", 0.0)) * count
            accumulators["seq_acc"] += float(group.get("seq_acc", 0.0)) * count
            accumulators["valid_ratio"] += float(group.get("valid_ratio", 0.0)) * count
            if group.get("similarity") is not None:
                accumulators["similarity"] += float(group["similarity"]) * count
                similarity_count += count

        if group_count == 0:
            continue
        grouped_metrics[group_name] = {
            "sample_count": group_count,
            "token_acc": accumulators["token_acc"] / group_count,
            "seq_acc": accumulators["seq_acc"] / group_count,
            "valid_ratio": accumulators["valid_ratio"] / group_count,
        }
        if similarity_count > 0:
            grouped_metrics[group_name]["similarity"] = accumulators["similarity"] / similarity_count

    return {
        "metrics": merged_metrics,
        "grouped_metrics": grouped_metrics,
        "inference_time": max(float(result.get("inference_time", 0.0)) for result in shard_results),
    }


def _json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_results(config, results, test_file_path, checkpoint_path, enabled_features):
    """Save metrics and examples in the existing output format."""
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    metrics = results["metrics"]

    saved_metrics = {
        "token_accuracy": float(metrics["token_accuracy"]),
        "sequence_accuracy": float(metrics["sequence_accuracy"]),
        "valid_smiles_ratio": float(metrics["valid_smiles_ratio"]),
        "tanimoto_similarity": (
            float(metrics["tanimoto_similarity"])
            if metrics.get("tanimoto_similarity") is not None
            else None
        ),
        "similarity_samples": int(metrics.get("similarity_samples", 0)),
        "total_samples": int(metrics["total_samples"]),
    }
    for key in (
        "teacher_forcing_token_accuracy",
        "teacher_forcing_sequence_accuracy",
        "topk_sequence_accuracy",
    ):
        if metrics.get(key) is not None:
            saved_metrics[key] = _json_safe(metrics[key])

    results_to_save = {
        "test_file": test_file_path,
        "checkpoint_path": checkpoint_path,
        "enabled_features": sorted(enabled_features),
        "metrics": saved_metrics,
        "grouped_metrics": _json_safe(results["grouped_metrics"]),
        "inference_time": float(results["inference_time"]),
    }
    if "parallel_world_size" in results:
        results_to_save["parallel_world_size"] = int(results["parallel_world_size"])

    results_path = os.path.join(config.SAVE_DIR, "test_results_ar.json")
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    examples_path = os.path.join(config.SAVE_DIR, "test_examples_ar.json")
    with open(examples_path, "w") as f:
        json.dump(_json_safe(metrics["examples"]), f, indent=2)
    logger.info(f"Examples saved to: {examples_path}")


def _resolve_parallel_world_size(requested_world_size=None):
    if not torch.cuda.is_available():
        raise RuntimeError("--parallel_eval requires CUDA")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 1:
        raise RuntimeError("--parallel_eval requires at least 2 visible CUDA devices")
    world_size = requested_world_size or visible_gpus
    if world_size <= 0:
        raise ValueError("--eval_world_size must be positive")
    if world_size > visible_gpus:
        raise ValueError(
            f"--eval_world_size={world_size} exceeds visible CUDA devices ({visible_gpus})"
        )
    return world_size


def _parallel_eval_worker(rank, world_size, worker_args, shard_output_dir):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    logger.info(f"Starting evaluation worker {rank + 1}/{world_size} on {device}")

    requested_features = parse_features(worker_args["features"])
    config = load_training_config(worker_args["config_path"], logger=logger)
    tokenizer = prepare_tokenizer(config, logger)
    model = load_model(
        config,
        tokenizer,
        worker_args["checkpoint_path"],
        device=device,
        use_data_parallel=False,
    )
    model_module = get_model_module(model)
    final_enabled_features = resolve_enabled_features(requested_features, model_module)

    test_file_path = worker_args["test_file"] or config.TEST_FILE
    test_loader = build_test_dataloader(
        config,
        tokenizer,
        test_file=test_file_path,
        enabled_features=final_enabled_features,
        shard_index=rank,
        num_shards=world_size,
    )
    results = evaluate_autoregressive_generation(
        model,
        test_loader,
        config,
        tokenizer,
        enabled_features=final_enabled_features,
    )
    results["enabled_features"] = sorted(final_enabled_features)
    results["test_file"] = test_file_path
    results["checkpoint_path"] = worker_args["checkpoint_path"]
    results["shard_index"] = rank
    results["num_shards"] = world_size

    shard_path = os.path.join(shard_output_dir, f"shard_{rank}.json")
    with open(shard_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    logger.info(f"Worker {rank + 1}/{world_size} wrote shard results to {shard_path}")


def run_parallel_evaluation(args, config, checkpoint_path):
    world_size = _resolve_parallel_world_size(args.eval_world_size)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    shard_output_dir = os.path.join(config.SAVE_DIR, "eval_shards", run_id)
    os.makedirs(shard_output_dir, exist_ok=True)
    logger.info(
        f"Running true multi-GPU evaluation with {world_size} workers. "
        f"Shard outputs: {shard_output_dir}"
    )

    worker_args = {
        "config_path": args.config_path,
        "checkpoint_path": checkpoint_path,
        "test_file": args.test_file,
        "features": args.features,
    }
    mp.spawn(
        _parallel_eval_worker,
        args=(world_size, worker_args, shard_output_dir),
        nprocs=world_size,
        join=True,
    )

    shard_results = []
    for rank in range(world_size):
        shard_path = os.path.join(shard_output_dir, f"shard_{rank}.json")
        with open(shard_path, "r") as f:
            shard_results.append(json.load(f))

    merged = merge_evaluation_results(shard_results)
    merged["parallel_world_size"] = world_size
    final_enabled_features = set(shard_results[0]["enabled_features"])
    test_file_path = shard_results[0]["test_file"]
    return merged, final_enabled_features, test_file_path


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test Spectra2Smiles-AR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with C-NMR + Formula
  python src/test.py --config_path configs/config.yaml --ckpt_path model.ckpt --features c_nmr,formula
  
  # Test with H-NMR only
  python src/test.py --config_path configs/config.yaml --ckpt_path model.ckpt --features h_nmr
  
  # Test with all features (based on model configuration)
  python src/test.py --config_path configs/config.yaml --ckpt_path model.ckpt
  
  # Test with specific test file and features
  python src/test.py --config_path configs/config.yaml --ckpt_path model.ckpt --test_file test.pkl.lz4 --features c_nmr,formula

  # True multi-GPU evaluation over sharded test-set samples
  CUDA_VISIBLE_DEVICES=0,1,2,3 python src/test.py --parallel_eval --config_path configs/config.yaml --ckpt_path model.ckpt --features c_nmr,formula
        """
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint file. If not provided, uses config.TEST_CKPT_PATH"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test dataset file (.pkl.lz4). "
             "If not provided, uses config.TEST_FILE from YAML/config_local.py. "
             "Example: --test_file /path/to/test.pkl.lz4"
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        type=str,
        default=None,
        help="Optional single YAML config path. If not provided, uses TrainingConfig/config_local.py.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of features to use. Options: c_nmr, h_nmr, formula. "
             "At least one NMR feature (c_nmr or h_nmr) must be specified. "
             "Example: --features c_nmr,formula or --features h_nmr,c_nmr. "
             "If not specified, uses all features based on model configuration."
    )
    parser.add_argument(
        "--parallel_eval",
        "--parallel-eval",
        action="store_true",
        help="Run true multi-GPU evaluation by sharding the test set across one process per GPU.",
    )
    parser.add_argument(
        "--eval_world_size",
        "--eval-world-size",
        type=int,
        default=None,
        help="Number of visible CUDA devices to use with --parallel_eval. Defaults to all visible devices.",
    )
    args = parser.parse_args()
    
    logger.info("===== Spectra2Smiles-AR Test Evaluation =====")
    
    # Parse features
    enabled_features = parse_features(args.features)
    if enabled_features:
        logger.info(f"Specified features: {enabled_features}")
    else:
        logger.info("No features specified, will use model configuration")
    
    # Load config and tokenizer
    config = load_training_config(args.config_path, logger=logger)
    tokenizer = prepare_tokenizer(config, logger)
    
    # Determine checkpoint path
    checkpoint_path = args.ckpt_path or getattr(config, "TEST_CKPT_PATH", None)
    if checkpoint_path is None:
        raise ValueError(
            "Checkpoint path not specified. Use:\n"
            "  1. Command line: --ckpt_path <path>\n"
            "  2. configs/config.yaml or config_local.py: TEST_CKPT_PATH = '<path>'"
        )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.parallel_eval:
        results, final_enabled_features, test_file_path = run_parallel_evaluation(
            args,
            config,
            checkpoint_path,
        )
        print_results(results, enabled_features=final_enabled_features)
        save_results(config, results, test_file_path, checkpoint_path, final_enabled_features)
        logger.info("\n===== Test Evaluation Complete =====")
        return
    
    # Load model
    model = load_model(config, tokenizer, checkpoint_path)
    
    # Get model module (handle DataParallel wrapper)
    model_module = get_model_module(model)
    
    # Log model configuration
    logger.info(f"\nModel configuration:")
    logger.info(f"  - C-NMR encoder: {'Enabled' if model_module.c_encoder is not None else 'Disabled'}")
    logger.info(f"  - H-NMR encoder: {'Enabled' if model_module.h_encoder is not None else 'Disabled'}")
    logger.info(f"  - Formula encoder: {'Enabled' if model_module.formula_encoder is not None else 'Disabled'}")
    
    # Determine final enabled features.
    final_enabled_features = resolve_enabled_features(enabled_features, model_module)
    if enabled_features is None:
        logger.info(f"Auto-detected features from model: {final_enabled_features}")
    else:
        logger.info(f"Using user-specified features: {final_enabled_features}")
    
    # Build test dataloader
    # Priority: 1. Command line argument 2. config_local.py 3. Default config
    test_file_path = args.test_file
    if test_file_path is None:
        test_file_path = config.TEST_FILE
        logger.info(f"Using TEST_FILE from config: {test_file_path}")
    else:
        logger.info(f"Using TEST_FILE from command line: {test_file_path}")
    
    test_loader = build_test_dataloader(
        config, 
        tokenizer, 
        test_file=test_file_path,
        enabled_features=final_enabled_features
    )
    
    # Evaluate
    results = evaluate_autoregressive_generation(
        model, 
        test_loader, 
        config, 
        tokenizer,
        enabled_features=final_enabled_features
    )
    
    # Print results
    print_results(results, enabled_features=final_enabled_features)
    save_results(config, results, test_file_path, checkpoint_path, final_enabled_features)
    
    logger.info("\n===== Test Evaluation Complete =====")

if __name__ == "__main__":
    main()
