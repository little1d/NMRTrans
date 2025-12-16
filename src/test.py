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
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

# Add src directory to path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import TrainingConfig, prepare_tokenizer
from data import MergedDataset
from model import NMR2SMILESModel

# Environment setup
os.environ["TF_DISABLE_MMAP"] = "1"
os.environ["TF_DISABLE_CUBLAS_TENSOR_OP_MATH"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")

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
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    
    # Disable RDKit error/warning messages to stderr
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.debug')
    
    RDKit_AVAILABLE = True
    logger.info("RDKit is available. SMILES validity and similarity will be checked.")
except ImportError:
    logger.warning("RDKit is not installed. Cannot validate SMILES or calculate similarity. Please run `pip install rdkit` to enable this feature.")
    RDKit_AVAILABLE = False
except AttributeError:
    # Older versions of RDKit might not have RDLogger
    RDKit_AVAILABLE = True
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
        
        # Generate Morgan fingerprints (radius=2, 2048 bits)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return float(similarity)
    
    except (ValueError, RuntimeError, AttributeError, TypeError):
        # Silently handle parsing errors
        return -1.0
    except Exception:
        # Catch any other exceptions
        return -1.0

def pad_peak_sequences(peak_sequences, max_peaks):
    """Pad peak sequences to the same length"""
    batch_size = len(peak_sequences)
    max_len = min(max(len(seq) for seq in peak_sequences), max_peaks)
    
    # Create padded tensor with shape [batch_size, max_peaks, 1]
    padded = torch.zeros(batch_size, max_peaks, 1)
    
    for i, peaks in enumerate(peak_sequences):
        num_peaks = min(len(peaks), max_peaks)
        if num_peaks > 0:
            padded[i, :num_peaks] = peaks[:num_peaks]
    
    return padded

def parse_chemical_formula(formula: str) -> dict:
    """Parse chemical formula string to atom counts dictionary."""
    if not formula or formula.strip() == "":
        return {}
    
    formula = formula.strip()
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    atom_counts = defaultdict(int)
    for atom, count in matches:
        count = int(count) if count else 1
        atom_counts[atom] += count
    
    return dict(atom_counts)

def parse_chemical_formula_to_vector(formula: str, atom_mapping: dict) -> torch.Tensor:
    """Convert chemical formula to atom count vector"""
    if not formula or formula.strip() == "":
        return torch.zeros(len(atom_mapping), dtype=torch.float)
    
    atoms = parse_chemical_formula(formula)
    vec = torch.zeros(len(atom_mapping), dtype=torch.float)
    for atom, count in atoms.items():
        if atom in atom_mapping:
            idx = atom_mapping[atom]
            vec[idx] = float(count)
    
    return vec

def peaks_collate_fn(batch, tokenizer, config, atom_mapping=None, enabled_features=None):
    """
    Collate function for NMR peak datasets
    
    Args:
        batch: List of samples
        tokenizer: Tokenizer instance
        config: TrainingConfig instance
        atom_mapping: Atom mapping for formula encoding (optional)
        enabled_features: Set of enabled features, e.g., {'c_nmr', 'h_nmr', 'formula'}
                        If None, processes all available features
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Default: process all features if not specified
    if enabled_features is None:
        enabled_features = {'c_nmr', 'h_nmr', 'formula'}
    
    # 1. Process SMILES
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
    
    # 2. Process spectrum data (only for enabled features)
    spectra_data = {}
    
    # H-NMR processing
    if 'h_nmr' in enabled_features and "h_nmr_peaks" in batch[0] and batch[0]["h_nmr_peaks"] is not None:
        h_peaks_list = []
        for item in batch:
            if item["h_nmr_peaks"] is not None and len(item["h_nmr_peaks"]) > 0:
                h_peaks = torch.tensor(item["h_nmr_peaks"], dtype=torch.float).unsqueeze(-1)
                h_peaks_list.append(h_peaks)
            else:
                h_peaks_list.append(torch.zeros((0, 1)))
        h_peaks_padded = pad_peak_sequences(h_peaks_list, config.MAX_PEAKS)
        spectra_data["h_nmr_peaks"] = h_peaks_padded
    
    # C-NMR processing
    if 'c_nmr' in enabled_features and "c_nmr_peaks" in batch[0] and batch[0]["c_nmr_peaks"] is not None:
        c_peaks_list = []
        for item in batch:
            if item["c_nmr_peaks"] is not None and len(item["c_nmr_peaks"]) > 0:
                c_peaks = torch.tensor(item["c_nmr_peaks"], dtype=torch.float).unsqueeze(-1)
                c_peaks_list.append(c_peaks)
            else:
                c_peaks_list.append(torch.zeros((0, 1)))
        c_peaks_padded = pad_peak_sequences(c_peaks_list, config.MAX_PEAKS)
        spectra_data["c_nmr_peaks"] = c_peaks_padded
    
    # Process formula vector
    if 'formula' in enabled_features and config.USE_FORMULA_GUIDANCE and atom_mapping is not None:
        formula_vectors = []
        for item in batch:
            # Handle missing molecular_formula field gracefully
            formula = item.get("molecular_formula", "")
            if formula is None:
                formula = ""
            vec = parse_chemical_formula_to_vector(formula, atom_mapping)
            formula_vectors.append(vec)
        formula_tensor = torch.stack(formula_vectors)
        spectra_data["formula_vector"] = formula_tensor
    
    return {
        "smiles": smiles_tensor,
        "original_smiles": original_smiles_list,
        **spectra_data
    }

def build_test_dataloader(config: TrainingConfig, tokenizer, test_file=None, enabled_features=None):
    """
    Build test data loader
    
    Args:
        config: TrainingConfig instance
        tokenizer: Tokenizer instance
        test_file: Path to test file (optional)
        enabled_features: Set of enabled features, e.g., {'c_nmr', 'formula'}
                         If None, uses all features based on model configuration
    """
    # Use provided test_file or fall back to config.TEST_FILE
    test_file_path = test_file or config.TEST_FILE
    logger.info(f"Loading test dataset from: {test_file_path}")
    
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    
    test_dataset = MergedDataset(test_file_path)
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Check data structure
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        has_source = "source" in sample
        has_formula = "molecular_formula" in sample
        has_h_nmr = "h_nmr_peaks" in sample
        has_c_nmr = "c_nmr_peaks" in sample
        logger.info(f"  - Has 'source' field: {has_source}")
        logger.info(f"  - Has 'molecular_formula' field: {has_formula}")
        logger.info(f"  - Has 'h_nmr_peaks' field: {has_h_nmr}")
        logger.info(f"  - Has 'c_nmr_peaks' field: {has_c_nmr}")
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_DATA_WORKERS if config.NUM_DATA_WORKERS > 0 else 0,
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

def load_model(config: TrainingConfig, tokenizer, checkpoint_path: str):
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
    
    if num_gpus > 1:
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
            h_peaks = batch.get("h_nmr_peaks") if 'h_nmr' in enabled_features else None
            formula_vector = batch.get("formula_vector") if 'formula' in enabled_features else None
            
            batch_size = smiles_ids.size(0)
            
            try:
                # ===== Teacher forcing 评估（与训练/validation 对齐）=====
                labels = smiles_ids.clone()
                labels[labels == config.PAD_TOKEN_ID] = -100
                tf_outputs = model_module(
                    c_peaks=c_peaks,
                    h_peaks=h_peaks,
                    formula_vector=formula_vector,
                    smiles_ids=labels,
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
                    h_peaks=h_peaks,
                    formula_vector=formula_vector,
                    max_length=config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS,
                    **gen_kwargs
                )
                
                # Calculate metrics for each sample
                for i in range(batch_size):
                    true_tokens = smiles_ids[i]
                    gen_tokens = generated_ids[i]
                    
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
                    
                    # Align to minimum content length
                    min_content_len = min(true_content_len, gen_content_len)
                    
                    if min_content_len > 0:
                        # Get content tokens (excluding special tokens)
                        true_content = true_tokens[valid_mask][:min_content_len]
                        gen_content = gen_tokens[gen_valid_mask][:min_content_len]
                        
                        # Token accuracy
                        correct = (true_content == gen_content)
                        token_acc = correct.sum().float() / min_content_len
                        
                        # Sequence accuracy (exact match of content)
                        seq_acc = correct.all().item()
                    else:
                        token_acc = torch.tensor(0.0)
                        seq_acc = 0.0
                    
                    # SMILES validity
                    is_valid = is_valid_smiles(gen_smiles)
                    
                    # Calculate Tanimoto similarity with ground truth
                    similarity = calculate_smiles_similarity(true_smiles, gen_smiles)
                    
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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test Spectra2Smiles-AR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with C-NMR + Formula
  python src/test.py --ckpt_path model.ckpt --features c_nmr,formula
  
  # Test with H-NMR only
  python src/test.py --ckpt_path model.ckpt --features h_nmr
  
  # Test with all features (based on model configuration)
  python src/test.py --ckpt_path model.ckpt
  
  # Test with specific test file and features
  python src/test.py --ckpt_path model.ckpt --test_file test.pkl.lz4 --features c_nmr,formula
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
             "If not provided, uses config.TEST_FILE (which can be set in config_local.py). "
             "Example: --test_file /path/to/test.pkl.lz4"
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
    args = parser.parse_args()
    
    logger.info("===== Spectra2Smiles-AR Test Evaluation =====")
    
    # Parse features
    enabled_features = parse_features(args.features)
    if enabled_features:
        logger.info(f"Specified features: {enabled_features}")
    else:
        logger.info("No features specified, will use model configuration")
    
    # Load config and tokenizer
    config = TrainingConfig()
    tokenizer = prepare_tokenizer(config, logger)
    
    # Determine checkpoint path
    checkpoint_path = args.ckpt_path or getattr(config, "TEST_CKPT_PATH", None)
    if checkpoint_path is None:
        raise ValueError(
            "Checkpoint path not specified. Use:\n"
            "  1. Command line: --ckpt_path <path>\n"
            "  2. config_local.py: TEST_CKPT_PATH = '<path>'"
        )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_model(config, tokenizer, checkpoint_path)
    
    # Get model module (handle DataParallel wrapper)
    model_module = get_model_module(model)
    
    # Log model configuration
    logger.info(f"\nModel configuration:")
    logger.info(f"  - C-NMR encoder: {'Enabled' if model_module.c_encoder is not None else 'Disabled'}")
    logger.info(f"  - H-NMR encoder: {'Enabled' if model_module.h_encoder is not None else 'Disabled'}")
    logger.info(f"  - Formula encoder: {'Enabled' if model_module.formula_encoder is not None else 'Disabled'}")
    
    # Determine final enabled features
    # If user specified features, validate against model configuration
    # If not specified, use model configuration
    final_enabled_features = enabled_features
    if final_enabled_features is None:
        # Auto-detect from model configuration
        final_enabled_features = set()
        if model_module.c_encoder is not None:
            final_enabled_features.add('c_nmr')
        if model_module.h_encoder is not None:
            final_enabled_features.add('h_nmr')
        if model_module.formula_encoder is not None:
            final_enabled_features.add('formula')
        logger.info(f"Auto-detected features from model: {final_enabled_features}")
    else:
        # Validate user-specified features against model
        if 'c_nmr' in final_enabled_features and model_module.c_encoder is None:
            raise ValueError(
                "Feature 'c_nmr' is requested but model does not have c_encoder. "
                "Please check your checkpoint or use different features."
            )
        if 'h_nmr' in final_enabled_features and model_module.h_encoder is None:
            raise ValueError(
                "Feature 'h_nmr' is requested but model does not have h_encoder. "
                "Please check your checkpoint or use different features."
            )
        if 'formula' in final_enabled_features and model_module.formula_encoder is None:
            raise ValueError(
                "Feature 'formula' is requested but model does not have formula_encoder. "
                "Please check your checkpoint or use different features."
            )
        # Ensure at least one NMR feature is enabled
        if not ('c_nmr' in final_enabled_features or 'h_nmr' in final_enabled_features):
            raise ValueError(
                "At least one NMR feature (c_nmr or h_nmr) must be enabled. "
                f"Specified features: {final_enabled_features}"
            )
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
    
    # Save results
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Save metrics (convert tensors to Python types)
    metrics = results["metrics"]
    results_to_save = {
        "test_file": test_file_path,  # Test dataset path
        "checkpoint_path": checkpoint_path,  # Model checkpoint path
        "enabled_features": list(final_enabled_features),  # Features used for evaluation
        "metrics": {
            "token_accuracy": float(metrics["token_accuracy"]),
            "sequence_accuracy": float(metrics["sequence_accuracy"]),
            "valid_smiles_ratio": float(metrics["valid_smiles_ratio"]),
            "tanimoto_similarity": float(metrics["tanimoto_similarity"]) if metrics.get("tanimoto_similarity") is not None else None,
            "similarity_samples": int(metrics.get("similarity_samples", 0)),
            "total_samples": int(metrics["total_samples"]),
        },
        "grouped_metrics": results["grouped_metrics"],
        "inference_time": float(results["inference_time"])
    }
    
    results_path = os.path.join(config.SAVE_DIR, "test_results_ar.json")
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Save examples
    examples_path = os.path.join(config.SAVE_DIR, "test_examples_ar.json")
    with open(examples_path, "w") as f:
        json.dump(metrics["examples"], f, indent=2)
    logger.info(f"Examples saved to: {examples_path}")
    
    logger.info("\n===== Test Evaluation Complete =====")

if __name__ == "__main__":
    main()
