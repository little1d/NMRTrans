#!/usr/bin/env python3
"""检查数据集的统计信息，诊断NaN问题"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pickle
import lz4.frame as lz4
import numpy as np

def check_dataset(file_path, num_samples=1000):
    """检查数据集的统计信息"""
    print(f"\n{'='*80}")
    print(f"Checking dataset: {file_path}")
    print(f"{'='*80}\n")
    
    # Load data
    if file_path.endswith(".lz4") or file_path.endswith(".pkl.lz4"):
        with lz4.open(file_path, "rb") as f:
            data = []
            while True:
                try:
                    batch = pickle.load(f)
                    if isinstance(batch, list):
                        data.extend(batch)
                    else:
                        data.append(batch)
                except EOFError:
                    break
    else:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Check first few samples
    c_values = []
    h_values = []
    
    for i, sample in enumerate(data[:num_samples]):
        if "c_nmr_peaks" in sample and sample["c_nmr_peaks"] is not None:
            c_values.extend(sample["c_nmr_peaks"])
        if "h_nmr_peaks" in sample and sample["h_nmr_peaks"] is not None:
            h_values.extend(sample["h_nmr_peaks"])
    
    # Statistics
    if c_values:
        c_array = np.array(c_values)
        print(f"\nC-NMR Peaks (first {num_samples} samples):")
        print(f"  Count: {len(c_values)}")
        print(f"  Range: [{c_array.min():.4f}, {c_array.max():.4f}]")
        print(f"  Mean: {c_array.mean():.4f}")
        print(f"  Std: {c_array.std():.4f}")
        print(f"  Has NaN: {np.isnan(c_array).any()}")
        print(f"  Has Inf: {np.isinf(c_array).any()}")
        
        # Check if negative values exist
        negative_count = (c_array < 0).sum()
        if negative_count > 0:
            print(f"  ⚠️  WARNING: {negative_count} negative values found!")
            print(f"     Negative range: [{c_array[c_array < 0].min():.4f}, {c_array[c_array < 0].max():.4f}]")
    
    if h_values:
        h_array = np.array(h_values)
        print(f"\nH-NMR Peaks (first {num_samples} samples):")
        print(f"  Count: {len(h_values)}")
        print(f"  Range: [{h_array.min():.4f}, {h_array.max():.4f}]")
        print(f"  Mean: {h_array.mean():.4f}")
        print(f"  Std: {h_array.std():.4f}")
        print(f"  Has NaN: {np.isnan(h_array).any()}")
        print(f"  Has Inf: {np.isinf(h_array).any()}")
        
        # Check if negative values exist
        negative_count = (h_array < 0).sum()
        if negative_count > 0:
            print(f"  ⚠️  WARNING: {negative_count} negative values found!")
            print(f"     Negative range: [{h_array[h_array < 0].min():.4f}, {h_array[h_array < 0].max():.4f}]")
    
    # Check SMILES
    smiles_lengths = []
    for sample in data[:num_samples]:
        if "original_smiles" in sample:
            smiles_lengths.append(len(sample["original_smiles"]))
    
    if smiles_lengths:
        print(f"\nSMILES Statistics (first {num_samples} samples):")
        print(f"  Length range: [{min(smiles_lengths)}, {max(smiles_lengths)}]")
        print(f"  Mean length: {np.mean(smiles_lengths):.2f}")
    
    print(f"\n{'='*80}\n")
    
    # Recommendations
    print("Recommendations:")
    if c_values and (c_array.max() > 500 or c_array.min() < -10):
        print("  ⚠️  C-NMR peaks have unusual range. Consider normalization.")
    if h_values and (h_array.max() > 500 or h_array.min() < -10):
        print("  ⚠️  H-NMR peaks have unusual range. Consider normalization.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    train_file = os.path.join(args.data_dir, "train.pkl.lz4")
    val_file = os.path.join(args.data_dir, "val.pkl.lz4")
    
    if os.path.exists(train_file):
        check_dataset(train_file, args.num_samples)
    if os.path.exists(val_file):
        check_dataset(val_file, args.num_samples)

