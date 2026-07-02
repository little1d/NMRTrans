import logging
import re
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import math
from collections import Counter
import json
logger = logging.getLogger(__name__)
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

MORGAN_GENERATOR_CHIRAL = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)

# ===========================================================
# 1. PeakEncoder —— 用于 H-NMR 与 C-NMR
# ===========================================================

class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout=0.1):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.ln2 = nn.LayerNorm(dim_V)

    def forward(self, Q, K, mask=None):
        Q = self.ln1(Q)
        K = self.ln1(K)
        q = self.fc_q(Q).view(Q.size(0), Q.size(1), self.num_heads, self.dim_V // self.num_heads)
        k = self.fc_k(K).view(K.size(0), K.size(1), self.num_heads, self.dim_V // self.num_heads)
        v = self.fc_v(K).view(K.size(0), K.size(1), self.num_heads, self.dim_V // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # (B, H, Lq, D/H)
        k = k.permute(0, 2, 3, 1)  # (B, H, D/H, Lk)
        v = v.permute(0, 2, 1, 3)  # (B, H, Lk, D/H)
        att = torch.matmul(q, k) / math.sqrt(self.dim_V // self.num_heads)
        if mask is not None:
            # mask: (B, Lk) -> (B, 1, 1, Lk)
            mask = mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v)  # (B, H, Lq, D/H)
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, Lq, H, D/H)
        out = out.view(out.size(0), out.size(1), self.dim_V)  # (B, Lq, D)
        out = self.fc_o(out)
        out = self.ln2(Q + self.dropout(out))
        return out

class SAB(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask)

class ISAB(nn.Module):
    """Induced Self-Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, dropout=0.1):
        super().__init__()
        self.num_inds = num_inds
        self.I = nn.Parameter(torch.randn(1, num_inds, dim_out))
        self.mab1 = MAB(dim_out, dim_in, dim_out, num_heads, dropout)
        self.mab2 = MAB(dim_in, dim_out, dim_out, num_heads, dropout)

    def forward(self, X, mask=None):
        batch_size = X.size(0)
        I = self.I.expand(batch_size, -1, -1)
        H = self.mab1(I, X, mask)  # (B, num_inds, dim_out)
        return self.mab2(X, H)     # (B, L, dim_out)

class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, dim, num_heads, num_seeds, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads, dropout)

    def forward(self, X, mask=None):
        batch_size = X.size(0)
        S = self.S.expand(batch_size, -1, -1)
        return self.mab(S, X, mask)

class SetTransformerPeakEncoder(nn.Module):
    """
    Set Transformer Encoder for NMR peaks with configurable number of layers
    输入: (B, L, 1) 的归一化ppm序列（包含重复，重复次数表示强度）
    输出: (B, L_unique, d_model) 的峰特征编码
    核心优势:
    - 天然支持无序集合输入
    - 置换不变性（permutation-invariant）
    - 支持任意层数的ISAB编码
    """
    def __init__(self, d_model=512, n_heads=8, num_inds=32, num_seeds=32,
                 n_layers=3, ff_dim=1024, dropout=0.1, max_peaks=60):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        self.n_layers = n_layers
        
        # 1. 输入投影层：将[ppm, intensity]投影到d_model维度
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 2. ISAB层（可配置层数）
        self.isab_layers = nn.ModuleList()
        for i in range(n_layers):
            self.isab_layers.append(
                ISAB(d_model, d_model, n_heads, num_inds, dropout)
            )
        
        # 3. PMA层（可选，用于提取全局表示）
        self.pma = PMA(d_model, n_heads, num_seeds, dropout)
        
        # 4. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 5. 用于padding的可学习token
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_model))

    def _process_peaks(self, peaks, mask):
        """
        将归一化后的peaks转换为(unique_ppm, intensities)
        peaks: (B, L, 1) - 归一化的ppm值，包含重复
        mask: (B, L) - 1表示有效峰，0表示padding
        返回:
        unique_ppm: (B, L_unique) - 唯一的归一化ppm值
        intensities: (B, L_unique) - 对应的强度（重复次数）
        valid_mask: (B, L_unique) - 1表示有效峰，0表示padding
        """
        batch_size, max_len, _ = peaks.shape
        device = peaks.device
        
        # 初始化输出张量
        max_unique_peaks = self.max_peaks
        unique_ppm = torch.zeros(batch_size, max_unique_peaks, device=device)
        intensities = torch.zeros(batch_size, max_unique_peaks, device=device)
        valid_mask = torch.zeros(batch_size, max_unique_peaks, device=device, dtype=torch.bool)
        
        for i in range(batch_size):
            # 获取当前样本的有效峰（根据mask）
            valid_indices = mask[i].bool()
            if not valid_indices.any():
                continue
            
            batch_peaks = peaks[i, valid_indices, 0]  # (num_valid_peaks,)
            
            # 将归一化ppm转换为字符串以便统计（避免浮点精度问题）
            rounded_peaks = torch.round(batch_peaks * 10000) / 10000.0
            
            # 统计每个ppm值的出现次数（强度）
            unique_vals, counts = torch.unique(rounded_peaks, return_counts=True, sorted=False)
            
            # 按ppm值排序（从大到小，符合NMR习惯）
            sorted_indices = torch.argsort(unique_vals, descending=True)
            unique_vals = unique_vals[sorted_indices]
            counts = counts[sorted_indices]
            
            # 限制最大峰数
            num_peaks_to_use = min(len(unique_vals), max_unique_peaks)
            
            # 填充unique_ppm和intensities
            unique_ppm[i, :num_peaks_to_use] = unique_vals[:num_peaks_to_use]
            intensities[i, :num_peaks_to_use] = counts[:num_peaks_to_use].float()
            valid_mask[i, :num_peaks_to_use] = True
        
        return unique_ppm, intensities, valid_mask

    def forward(self, peaks, mask=None):
        """
        peaks: (B, L, 1) - 归一化的ppm序列，包含重复（重复次数=强度）
        mask: (B, L) - 1表示有效峰，0表示padding
        返回: (B, L_unique, d_model) - 峰特征编码
        """
        if peaks is None:
            return None

        if mask is None:
            # 如果没有提供mask，根据peaks是否为0创建mask
            mask = (peaks.abs() > 1e-6).all(-1).long()
        
        # 1. 预处理：将归一化peaks转换为(unique_ppm, intensities)对
        unique_ppm, intensities, valid_mask = self._process_peaks(peaks, mask)
        # unique_ppm shape: (B, L_unique) - 已归一化的ppm值 [0,1]
        # intensities shape: (B, L_unique) - 强度值
        
        batch_size, num_unique_peaks = unique_ppm.shape
        
        # 2. 创建输入特征: [ppm, intensity]
        features = torch.stack([unique_ppm, intensities], dim=-1)  # (B, L_unique, 2)
        
        # 3. 输入投影
        X = self.input_proj(features)  # (B, L_unique, d_model)
        
        # 4. 应用mask：将padding位置替换为pad_token
        pad_mask = ~valid_mask  # True表示需要padding的位置
        
        # 5. Set Transformer编码（多层ISAB）
        for i, isab_layer in enumerate(self.isab_layers):
            X = isab_layer(X, mask=pad_mask if pad_mask.any() else None)
        
        # 6. （可选）PMA获取全局表示
        global_repr = None
        global_repr = self.pma(X, mask=pad_mask if pad_mask.any() else None)  # (B, num_seeds, d_model)
        
        # 7. 输出投影
        X = self.output_proj(X)  # (B, L_unique, d_model)
        
        return X, global_repr, valid_mask

class SetTransformer1HNMRPeakEncoder(nn.Module):
    """
    Specialized Set Transformer Encoder for 1HNMR peaks with 5 features:
    1. chemical_shift (numeric, ppm)
    2. peak_width     (numeric, ppm)  ← NEW: width/range of complex peaks
    3. split_pattern  (discrete token, embedded)
    4. integral       (numeric, H count)
    5. J-coupling     (6 numeric values, padded)
    
    Input format for each peak: [chem_shift, peak_width, split_idx, integral_value, j1, j2, j3, j4, j5, j6]
    Total input dimension: 10 features (2 numeric + 1 embedded + 1 numeric + 6 numeric)
    """
    def __init__(self, d_model=512, n_heads=8, num_inds=32, num_seeds=32,
                 n_layers=3, ff_dim=1024, dropout=0.1, max_peaks=60,
                 split_vocab_size=16):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        self.n_layers = n_layers
        self.split_vocab_size = split_vocab_size
        
        # 1. Split pattern embeddings
        self.split_embedding = nn.Embedding(split_vocab_size, d_model // 4)
        
        # 2. Input projection for continuous features
        # ✅ UPDATED: Continuous features = chem_shift(1) + width(1) + integral(1) + J(6) = 9 features
        self.continuous_proj = nn.Sequential(
            nn.Linear(9, d_model // 2),  # ← 8 → 9
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 3. Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),  # continuous + split embedding
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 4. ISAB layers
        self.isab_layers = nn.ModuleList()
        for i in range(n_layers):
            self.isab_layers.append(
                ISAB(d_model, d_model, n_heads, num_inds, dropout)
            )
        
        # 5. PMA layer
        self.pma = PMA(d_model, n_heads, num_seeds, dropout)
        
        # 6. Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 7. Padding token
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_model))

    def shuffle_peaks(self, peaks, mask=None):
        """
        仅在 mask=1 的有效位置内打乱 peaks，mask 本身保持不变
        """
        B, L, D = peaks.shape
        shuffled_peaks = peaks.clone()
        
        for i in range(B):
            valid_idx = torch.where(mask[i])[0]
            k = len(valid_idx)
            if k > 1:
                perm = torch.randperm(k, device=peaks.device)
                shuffled_idx = valid_idx[perm]
                shuffled_peaks[i, valid_idx] = peaks[i, shuffled_idx]
        return shuffled_peaks
    
    def forward(self, peaks, mask=None):
        """
        peaks: (B, L, 10) - 1HNMR features [chem_shift, width, split_idx, integral, j1-j6]
        mask:  (B, L)     - 1=valid peak, 0=padding
        Returns: 
            X: (B, L, d_model) - per-peak encoded features
            global_repr: (B, num_seeds, d_model) - global representation via PMA
            mask: (B, L) - boolean mask
        """
        if peaks is None:
            return None, None, None

        if mask is None:
            mask = (peaks.abs().sum(-1) > 1e-6).long()
        
        # Optional: shuffle peaks within valid positions (for augmentation)
        # peaks = self.shuffle_peaks(peaks, mask)
        
        batch_size, seq_len, feat_dim = peaks.shape
        device = peaks.device
        
        # ✅ UPDATED: Separate 10-dim features
        chem_shift = peaks[..., 0:1]    # (B, L, 1) - index 0
        peak_width = peaks[..., 1:2]    # (B, L, 1) - index 1 ← NEW
        split_idx = peaks[..., 2].long()  # (B, L)   - index 2 (was 1)
        integral = peaks[..., 3:4]      # (B, L, 1) - index 3 (was 2)
        j_coupling = peaks[..., 4:10]   # (B, L, 6) - indices 4-9 (was 3-8)
        
        # ✅ UPDATED: Concatenate continuous features (9 total)
        continuous_features = torch.cat([
            chem_shift,    # 1
            peak_width,    # 1 ← NEW
            integral,      # 1
            j_coupling     # 6
        ], dim=-1)  # (B, L, 9)
        
        # Project continuous features
        continuous_proj = self.continuous_proj(continuous_features)  # (B, L, d_model)
        
        # Process split pattern features
        split_embed = self.split_embedding(split_idx)  # (B, L, d_model//4)
        
        # Fuse features
        fused_features = torch.cat([continuous_proj, split_embed], dim=-1)  # (B, L, 5d_model/4)
        X = self.feature_fusion(fused_features)  # (B, L, d_model)
        
        # Apply mask
        pad_mask = ~mask.bool()
        if pad_mask.any():
            pad_tokens = self.pad_token.expand(batch_size, seq_len, -1)
            X = torch.where(pad_mask.unsqueeze(-1), pad_tokens, X)
        
        # Set Transformer encoding
        for isab_layer in self.isab_layers:
            X = isab_layer(X, mask=pad_mask if pad_mask.any() else None)
        
        # PMA for global representation
        global_repr = self.pma(X, mask=pad_mask if pad_mask.any() else None)  # (B, num_seeds, d_model)
        
        # Output projection
        X = self.output_proj(X)  # (B, L, d_model)
        
        return X, global_repr, mask.bool()

# ===========================================================
# 2. Multi-modal Fusion Layer —— 合并 CNMR + HNMR + guidance
# ===========================================================

class FusionLayer(nn.Module):
    """
    融合层，同时返回特征和对应的attention mask
    支持多种输入组合：
    1. 仅局部特征 (z_c, z_h)
    2. 仅全局特征 (global_c, global_h)
    3. 仅formula指导 (z_guidance)
    4. 任意组合
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model

    def forward(self, z_c=None, z_h=None, global_c=None, global_h=None, z_guidance=None,
                c_mask=None, h_mask=None):
        """
        Args:
            z_c: (B, L_c, d_model) - C-NMR特征（可选）
            z_h: (B, L_h, d_model) - H-NMR特征（可选）
            global_c: (B, num_seeds_c, d_model) - C-NMR全局特征（可选）
            global_h: (B, num_seeds_h, d_model) - H-NMR全局特征（可选）
            z_guidance: (B, 1, d_model) - 化学式指导特征（可选）
            c_mask: (B, L_c) - C-NMR的mask，1表示有效，0表示padding（可选）
            h_mask: (B, L_h) - H-NMR的mask，1表示有效，0表示padding（可选）
        Returns:
            z_all: (B, L_total, d_model) - 融合后的特征
            attention_mask: (B, L_total) - 对应的attention mask，1表示有效，0表示padding
        """
        # 检查是否有任何有效输入
        has_local_features = (z_c is not None) or (z_h is not None)
        has_global_features = (global_c is not None) or (global_h is not None)
        has_guidance = z_guidance is not None
        
        if not (has_local_features or has_global_features or has_guidance):
            raise ValueError("No input features provided! Need at least one of: "
                            "local features (z_c/z_h), global features (global_c/global_h), or guidance (z_guidance)")
        
        all_features = []
        all_masks = []
        batch_size = None
        device = None
        
        # 确定batch_size和device
        for tensor in [z_c, z_h, global_c, global_h, z_guidance]:
            if tensor is not None:
                batch_size = tensor.size(0)
                device = tensor.device
                break
        
        if batch_size is None:
            raise ValueError("Cannot determine batch size from inputs")
        
        # 1. 处理局部特征 (z_c, z_h)
        if z_c is not None:
            all_features.append(z_c)
            if c_mask is not None:
                all_masks.append(c_mask)
            else:
                all_masks.append(torch.ones(z_c.shape[:2], dtype=torch.long, device=device))
        
        if z_h is not None:
            all_features.append(z_h)
            if h_mask is not None:
                all_masks.append(h_mask)
            else:
                all_masks.append(torch.ones(z_h.shape[:2], dtype=torch.long, device=device))
        
        # 2. 处理全局特征 (global_c, global_h)
        if global_c is not None:
            all_features.append(global_c)
            # 全局特征通常都是有效的
            all_masks.append(torch.ones(global_c.shape[:2], dtype=torch.long, device=device))
        
        if global_h is not None:
            all_features.append(global_h)
            all_masks.append(torch.ones(global_h.shape[:2], dtype=torch.long, device=device))
        
        # 3. 处理化学式指导 (z_guidance)
        if z_guidance is not None:
            all_features.append(z_guidance)
            # 化学式guidance通常是有效的
            guidance_mask = torch.ones((batch_size, z_guidance.size(1)),
                                     dtype=torch.long, device=device)
            all_masks.append(guidance_mask)
        
        # 4. 拼接所有特征和mask
        if all_features:
            z_all = torch.cat(all_features, dim=1)
            attention_mask = torch.cat(all_masks, dim=1) if all_masks else None
        else:
            # 理论上不会到达这里，因为前面已经检查过
            z_all = torch.zeros(batch_size, 0, self.d_model, device=device)
            attention_mask = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        
        # 5. 确保attention_mask存在（如果没有任何mask，创建全1的mask）
        if attention_mask is None or attention_mask.shape[1] == 0:
            attention_mask = torch.ones((batch_size, z_all.size(1)),
                                       dtype=torch.long, device=device)
        
        # 6. 验证输出形状
        assert z_all.size(0) == batch_size, f"Batch size mismatch: {z_all.size(0)} vs {batch_size}"
        assert attention_mask.size(0) == batch_size, f"Mask batch size mismatch: {attention_mask.size(0)} vs {batch_size}"
        assert z_all.size(1) == attention_mask.size(1), \
            f"Feature and mask length mismatch: {z_all.size(1)} vs {attention_mask.size(1)}"
        
        return z_all, attention_mask

# ===========================================================
# 3. FormulaEncoder —— 将化学式向量转换为guidance embedding
# ===========================================================

class FormulaEncoder(nn.Module):
    def __init__(self, formula_vector_size, d_model=512, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(formula_vector_size, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, formula_vector):
        if formula_vector is None:
            return None
        x = self.net(formula_vector)  # (B, d_model)
        return x.unsqueeze(1)  # (B, 1, d_model)

# ===========================================================
# 4. 主模型：NMR → SMILES（T5 decoder-only）
# ===========================================================

class NMR2SMILESModel(pl.LightningModule):
    """
    主架构：
    - 不使用 T5 encoder
    - 自己实现 H/C encoder
    - 最终输出喂给 T5 decoder
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        d_model = config.PEAK_ENCODER_D_MODEL
        
        # 消融实验：根据配置决定是否初始化 C/H encoder
        use_c_nmr = getattr(config, "USE_C_NMR", True)
        use_h_nmr = getattr(config, "USE_H_NMR", True)
        use_formula = getattr(config, "USE_FORMULA_GUIDANCE", True)
        
        # 验证：至少需要启用一个NMR模态（C或H），Formula是可选的
        if not (use_c_nmr or use_h_nmr):
            raise ValueError("至少需要启用一个NMR模态：USE_C_NMR 或 USE_H_NMR（Formula是可选的）")
        
        # 记录消融实验配置
        logger.info("\n" + "="*80)
        logger.info("消融实验配置:")
        logger.info(f"  USE_C_NMR: {use_c_nmr}")
        logger.info(f"  USE_H_NMR: {use_h_nmr}")
        logger.info(f"  USE_FORMULA_GUIDANCE: {use_formula}")
        
        # 显示当前配置对应的组合
        modalities = []
        if use_c_nmr:
            modalities.append("C")
        if use_h_nmr:
            modalities.append("H")
        if use_formula:
            modalities.append("Formula")
        logger.info(f"  当前组合: {'+'.join(modalities)}")
        logger.info("="*80)
        
        # 定义split词汇表
        self.split_vocab = [
        '<unk>', 'm', 'd', 's', 'dd', 't', 'ddd', 'q',
        'dt', 'td', 'br', 'ddt', 'dq', 'tt', 'quint',
        'dddd', 'qd', 'sept', 'ddp', 'ddq', 'bd', 'dqd',
        ]
        self.split_to_idx = {token: idx for idx, token in enumerate(self.split_vocab)}
        
        # 主模态 encoder（根据配置初始化）
        if use_c_nmr:
            self.c_encoder = SetTransformerPeakEncoder(
            d_model=d_model,
                num_inds=32,
                num_seeds=4,
            n_layers=config.PEAK_ENCODER_N_LAYERS,
            n_heads=config.PEAK_ENCODER_N_HEADS,
                ff_dim=config.PEAK_ENCODER_FF_DIM,
                dropout=config.PEAK_ENCODER_DROPOUT,
                max_peaks=config.MAX_PEAKS
            )
            logger.info("✅ C-NMR encoder 已启用")
        else:
            self.c_encoder = None
            logger.info("❌ C-NMR encoder 已禁用（消融实验）")
        
        if use_h_nmr:
            self.h_encoder = SetTransformer1HNMRPeakEncoder(
            d_model=d_model,
                num_inds=32,
                num_seeds=4,
            n_layers=config.PEAK_ENCODER_N_LAYERS,
            n_heads=config.PEAK_ENCODER_N_HEADS,
                ff_dim=config.PEAK_ENCODER_FF_DIM,
                dropout=config.PEAK_ENCODER_DROPOUT,
                max_peaks=config.MAX_PEAKS,
                split_vocab_size=len(self.split_vocab)
            )
            logger.info("✅ H-NMR encoder (with features) 已启用")
        else:
            self.h_encoder = None
            logger.info("❌ H-NMR encoder 已禁用（消融实验）")
        
        # 添加FormulaEncoder（如果配置中启用了化学式指导）
        if getattr(config, "USE_FORMULA_GUIDANCE", True):
            if not hasattr(config, "ALL_ATOMS"):
                raise ValueError("config.ALL_ATOMS must be set when using formula guidance")
            self.formula_encoder = FormulaEncoder(
                formula_vector_size=len(config.ALL_ATOMS),
                d_model=d_model,
            )
            logger.info(f"Formula encoder initialized with {len(config.ALL_ATOMS)} atoms")
        else:
            self.formula_encoder = None

        # Fusion
        self.fusion = FusionLayer(d_model=d_model)

        # T5 用于 decoder-only（但依然 load 整个模型）
        # Note: This loads the T5 architecture. Actual weights will be loaded from checkpoint.
        logger.debug(f"Initializing T5 model architecture from: {config.T5_MODEL_NAME}")
        if getattr(config, "USE_RANDOM_T5_INIT", False):
            # 使用 T5Config 随机初始化（不加载预训练权重）
            t5_config = T5Config.from_pretrained(
                config.T5_MODEL_NAME,
                local_files_only=True
            )
            t5_config.vocab_size = len(tokenizer)
            self.t5 = T5ForConditionalGeneration(t5_config)
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                config.T5_MODEL_NAME,
                local_files_only=True  # Use local model files only, don't access network
            )
        
        # 对齐 T5 的词表大小与自定义 SMILES tokenizer，确保 embedding / lm_head 尺寸一致
        self._resize_t5_embeddings_to_tokenizer()
        
        if getattr(config, "REMOVE_CROSS_ATTENTION_POSITION_BIAS", False):
            for layer in self.t5.decoder.block:
                cross_attention = layer.layer[1].EncDecAttention
                # 禁用位置偏置
                cross_attention.has_relative_attention_bias = False
                cross_attention.relative_attention_bias = None
                # 可选：验证是否成功移除
                print(f"Cross-attention position bias disabled: {cross_attention.has_relative_attention_bias}")
        
        # Log final embedding/LM head shapes for sanity check
        embed_shape = tuple(self.t5.get_input_embeddings().weight.shape)
        lm_head_shape = tuple(self.t5.lm_head.weight.shape)
        logger.info(f"T5 embedding shape: {embed_shape}, LM head shape: {lm_head_shape}")
        
        # Set T5 to training mode
        self.t5.train()
        
        # Optionally freeze T5 decoder
        if config.FREEZE_T5_DECODER:
            logger.info("Freezing T5 decoder parameters")
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
        
        # Even if frozen, keep in train mode for other modules
        self.t5.train()
        
        # For logging
        self.validation_outputs = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['tokenizer'])

    def _pad_j_coupling(self, j_coupling_list, max_j=6):
        """Pad J-coupling list to fixed length"""
        if not isinstance(j_coupling_list, list):
            j_coupling_list = []
        
        padded = [0.0] * max_j
        for i, val in enumerate(j_coupling_list[:max_j]):
            try:
                padded[i] = float(val)
            except (TypeError, ValueError):
                padded[i] = 0.0
        return padded

    def _prepare_h_nmr_features(self, tokenized_input_str):
        """Prepare 1HNMR features from tokenized_input string"""
        try:
            tokenized_input = json.loads(tokenized_input_str)
            h_nmr_data = tokenized_input.get("1HNMR", [])
            
            features = []
            for peak in h_nmr_data:
                if len(peak) < 4:
                    continue
                
                # 1. chemical_shift (numeric)
                chem_shift = float(peak[0])
                
                # 2. split pattern (discrete token)
                split_str = str(peak[2]).strip().lower() if len(peak) > 2 else "<unk>"
                split_idx = self.split_to_idx.get(split_str, self.split_to_idx["<unk>"])
                
                # 3. integral (numeric)
                integral_str = str(peak[3]).strip() if len(peak) > 3 else '1H'
                # Extract number before 'H'
                integral_value = 1.0
                match = re.search(r'(\d+)(?:H|h)?', integral_str)
                if match:
                    integral_value = float(match.group(1))
                
                # 4. J-coupling (6 numeric values, padded)
                j_coupling = peak[4] if len(peak) > 4 and isinstance(peak[4], list) else []
                padded_j = self._pad_j_coupling(j_coupling)
                
                # Combine all features: [chem_shift, split_idx, integral_value, j1-j6]
                peak_features = [chem_shift, split_idx, integral_value] + padded_j
                features.append(peak_features)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error preparing H-NMR features: {str(e)}")
            return []

    def forward(
        self,
        c_peaks=None,
        h_peaks=None,
        formula=None,
        smiles_ids=None,
        attention_mask=None,
        formula_vector=None,
        c_nmr_mask=None,
        h_nmr_mask=None,
        h_features=None,  # New: preprocessed 1HNMR features
        **kwargs
    ):
        """
        输入:
          c_peaks: List[List[float]] → 已 padded tensor
          h_peaks: 同上
        h_features: preprocessed 1HNMR features tensor
          smiles_ids: tokenized smiles (labels)
        """
        # 使用 PeakEncoder + Fusion（原始模式）
        # C-NMR（根据配置和输入决定）
        z_c = None
        global_c = None
        if self.c_encoder is not None and c_peaks is not None:
            z_c, global_c, c_nmr_mask = self.c_encoder(c_peaks, mask=c_nmr_mask)
        
        # H-NMR（根据配置和输入决定）- 使用新特征
        z_h = None
        global_h = None
        if self.h_encoder is not None and h_features is not None:
            z_h, global_h, h_nmr_mask = self.h_encoder(h_features, mask=h_nmr_mask)
        
        # 新增：处理化学式guidance
        z_guidance = None
        if self.formula_encoder is not None and formula_vector is not None:
            z_guidance = self.formula_encoder(formula_vector)  # (B, 1, d_model)
        
        # global_c = None
        # global_h = None
        # 融合 - 现在返回特征和对应的attention mask
        encoder_hidden, attention_mask = self.fusion(
            z_c, z_h,
            global_c, global_h,
            z_guidance,
            c_mask=c_nmr_mask,
            h_mask=h_nmr_mask
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        # 验证mask形状是否匹配
        batch_size, seq_len, _ = encoder_hidden.shape
        assert attention_mask.shape == (batch_size, seq_len), \
            f"Mask shape {attention_mask.shape} doesn't match sequence length {seq_len}"

        # 送入 T5 decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,  # 这里使用1表示有效，0表示padding
            labels=smiles_ids,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step for autoregressive generation."""
        smiles_ids = batch["smiles"].long()
        
        # 根据配置获取输入（消融实验）
        c_peaks = batch.get("c_nmr_peaks") if self.c_encoder is not None else None
        c_nmr_mask = batch.get("c_nmr_mask") if self.c_encoder is not None else None
        
        # 获取预处理的1HNMR特征
        h_features = batch.get("h_nmr_features") if self.h_encoder is not None else None
        h_nmr_mask = batch.get("h_nmr_mask") if self.h_encoder is not None else None
        
        # Check for NaN in input data
        if c_peaks is not None and torch.isnan(c_peaks).any():
            logger.warning(f"NaN detected in c_peaks at batch {batch_idx}")
        if h_features is not None and torch.isnan(h_features).any():
            logger.warning(f"NaN detected in h_features at batch {batch_idx}")

        # if self.training:
        #     # C-NMR: 归一化空间加噪声 (约 1-2 ppm 等效)
        #     if c_peaks is not None and c_nmr_mask is not None:
        #         noise = torch.randn_like(c_peaks) * 0.005  # ✅ 归一化尺度
        #         c_peaks = c_peaks + noise
        #         c_peaks = torch.clamp(c_peaks, 0.0, 1.0)  # 保持范围
            
        #     # H-NMR: 化学位移加噪声 (约 0.1 ppm 等效)
        #     if h_features is not None and h_nmr_mask is not None:
        #         h_shift_noise = torch.randn_like(h_features[:, :, 0:1]) * 0.08  # ✅ 归一化尺度
        #         h_features = h_features.clone()  # 避免 inplace 操作
        #         h_features[:, :, 0:1] = h_features[:, :, 0:1] + h_shift_noise
        #         h_features[:, :, 0:1] = torch.clamp(h_features[:, :, 0:1], 0.0, 1.0)
                
        # T5 expects labels with -100 for positions to ignore
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None
        
        # Forward pass
        outputs = self(
            c_peaks=c_peaks,
            h_peaks=None,  # Not used directly anymore
            formula_vector=formula_vector,
            smiles_ids=labels,
            c_nmr_mask=c_nmr_mask,
            h_nmr_mask=h_nmr_mask,
            h_features=h_features,  # Pass preprocessed features
        )
        
        loss = outputs.loss
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf loss detected at batch {batch_idx}")
            logger.error(f"C peaks shape: {c_peaks.shape if c_peaks is not None else None}")
            logger.error(f"H features shape: {h_features.shape if h_features is not None else None}")
            logger.error(f"SMILES ids shape: {smiles_ids.shape}")
            logger.error(f"Labels shape: {labels.shape}")
            # Check data range
            if c_peaks is not None:
                logger.error(f"C peaks range: [{c_peaks.min():.4f}, {c_peaks.max():.4f}]")
            if h_features is not None:
                logger.error(f"H features range: [{h_features.min():.4f}, {h_features.max():.4f}]")
            # Check if logits have NaN
            if torch.isnan(outputs.logits).any():
                logger.error("NaN detected in model logits!")
            # Skip this batch
            return None
        
        # Compute accuracy
        with torch.no_grad():
            logits = outputs.logits
            pred_tokens = logits.argmax(dim=-1)
            # Only consider non-padding positions
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float()
            
            # Sequence accuracy: all tokens correct
            seq_correct = ((pred_tokens == smiles_ids) | ~valid_mask).all(dim=1)
            seq_acc = seq_correct.float().mean()
        
        # Log metrics (添加 logger=True 以同步到 SwanLab)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, logger=True)
        self.log("train_token_acc", token_acc, prog_bar=True, sync_dist=True, logger=True)
        self.log("train_seq_acc", seq_acc, prog_bar=True, sync_dist=True, logger=True)
        
        # Log learning rate
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=False, logger=True)
        
        return loss
    
    @staticmethod
    def evaluate_smiles_pair(pred_smiles, origin_smiles):
        """
        Evaluate a single pair of SMILES using RDKit.
        Returns: (accuracy, valid, similarity)
        """
        acc = 0.0
        valid = 0.0
        similarity = 0.0
        pred_mol = None
        try:
            pred_mol = Chem.MolFromSmiles(pred_smiles)
        except Exception:
            pass # pred_mol remains None
        
        if pred_mol is None:
            return acc, valid, similarity
        
        # Valid molecule
        valid = 1.0
        
        try:
            origin_mol = Chem.MolFromSmiles(origin_smiles)
            if origin_mol is None:
                # Ground truth is invalid? Should not happen usually.
                return acc, valid, similarity
            
            # Exact Match (Canonical SMILES check)
            if Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(origin_mol):
                acc = 1.0
            
            # Tanimoto Similarity
            fp_1 = MORGAN_GENERATOR_CHIRAL.GetFingerprint(pred_mol)
            fp_2 = MORGAN_GENERATOR_CHIRAL.GetFingerprint(origin_mol)
            similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
        except Exception as e:
            # logger.warning(f"RDKit evaluation error: {e}")
            pass
        
        return acc, valid, similarity
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with autoregressive generation and RDKit evaluation.
        ✅ 修复：对整个 batch 进行生成评估，而不是只取第 1 个样本
        """
        if batch_idx == 0 and self.global_rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting validation at epoch {self.current_epoch}")
            logger.info(f"{'='*80}")
        
        smiles_ids = batch["smiles"].long()
        original_smiles_list = batch["original_smiles"]
        batch_size = smiles_ids.size(0)
        
        # 根据配置获取输入（消融实验）
        c_peaks = batch.get("c_nmr_peaks") if self.c_encoder is not None else None
        c_nmr_mask = batch.get("c_nmr_mask") if self.c_encoder is not None else None
        h_features = batch.get("h_nmr_features") if self.h_encoder is not None else None
        h_nmr_mask = batch.get("h_nmr_mask") if self.h_encoder is not None else None
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None
        
        # ========== Teacher Forcing 评估（Token 级别）==========
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        with torch.no_grad():
            outputs = self(
                c_peaks=c_peaks,
                h_features=h_features,
                formula_vector=formula_vector,
                smiles_ids=labels,
                c_nmr_mask=c_nmr_mask,
                h_nmr_mask=h_nmr_mask,
            )
            logits = outputs.logits
            pred_tokens = logits.argmax(dim=-1)
            
            # Token accuracy
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float() if valid_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
            self.log("val_token_acc", token_acc, prog_bar=True, sync_dist=True, logger=True)
        
        # ========== 自回归生成评估（整个 Batch）==========
        val_mol_acc = torch.tensor(0.0, device=self.device)
        val_validity = torch.tensor(0.0, device=self.device)
        val_similarity = torch.tensor(0.0, device=self.device)
        seq_acc = torch.tensor(0.0, device=self.device)
        samples_evaluated = 0
        
        try:
            # ✅ 修复：对整个 batch 进行生成，不是 [:1]
            generated_ids = self.generate(
                c_peaks=c_peaks,
                h_features=h_features,
                formula_vector=formula_vector,
                c_nmr_mask=c_nmr_mask,
                h_nmr_mask=h_nmr_mask,
                max_length=self.config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS,
                num_beams=1,
                do_sample=False
            )
            
            acc_list, valid_list, sim_list = [], [], []
            rdBase.DisableLog('rdApp.error')
            try:
                # ✅ 修复：遍历整个 batch，不是只评估第 1 个
                for i in range(batch_size):
                    if i >= len(original_smiles_list):
                        break
                    
                    pred_smiles = self.tokens_to_smiles(generated_ids[i])
                    true_smiles = original_smiles_list[i]
                    acc, valid, sim = self.evaluate_smiles_pair(pred_smiles, true_smiles)
                    
                    acc_list.append(acc)
                    valid_list.append(valid)
                    if sim >= 0:
                        sim_list.append(sim)
                
                samples_evaluated = len(acc_list)
                
                if len(acc_list) > 0:
                    val_mol_acc = torch.tensor(acc_list, device=self.device).float().mean()
                    val_validity = torch.tensor(valid_list, device=self.device).float().mean()
                    seq_acc = val_mol_acc  # RDKit exact match
                    
                    if len(sim_list) > 0:
                        val_similarity = torch.tensor(sim_list, device=self.device).float().mean()
                
            finally:
                rdBase.EnableLog('rdApp.error')
            
            # ✅ 记录评估样本数量（用于调试）
            self.log("val_samples_evaluated", float(samples_evaluated), prog_bar=False, logger=True)
            
            # ✅ 记录评估失败率（用于调试）
            if batch_size > 0:
                failure_rate = 1.0 - (samples_evaluated / batch_size)
                self.log("val_eval_failure_rate", failure_rate, prog_bar=False, logger=True)
            
        except Exception as e:
            logger.warning(f"Validation generation/evaluation failed at batch {batch_idx}: {e}")
            # 保持默认值 0.0
        
        # ========== 记录指标 ==========
        self.log("val_seq_acc", seq_acc, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_validity", val_validity, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_similarity", val_similarity, prog_bar=True, sync_dist=True, logger=True)
        
        # ========== 记录示例（每 10 个 batch 记录一次）==========
        if batch_idx % 10 == 0 and self.global_rank == 0 and samples_evaluated > 0:
            try:
                # 记录前 3 个样本的生成结果
                for i in range(min(3, samples_evaluated)):
                    pred_smiles = self.tokens_to_smiles(generated_ids[i])
                    true_smiles = original_smiles_list[i]
                    logger.info(f"\n[Validation Example {batch_idx}-{i}]")
                    logger.info(f"Original:  {true_smiles}")
                    logger.info(f"Generated: {pred_smiles}")
                    logger.info(f"Match:     {pred_smiles == true_smiles}")
            except Exception as e:
                logger.warning(f"Failed to log validation example: {e}")
        
        # ========== 保存验证输出（用于 on_validation_epoch_end 聚合）==========
        self.validation_outputs.append({
            "val_token_acc": token_acc.item(),
            "val_seq_acc": seq_acc.item(),
            "val_validity": val_validity.item(),
            "val_similarity": val_similarity.item(),
            "samples_evaluated": samples_evaluated,
            "batch_size": batch_size,
        })
        
        return {
            "val_token_acc": token_acc,
            "val_seq_acc": seq_acc,
            "val_validity": val_validity,
            "val_similarity": val_similarity,
            "samples_evaluated": torch.tensor(samples_evaluated, device=self.device),
            "batch_size": torch.tensor(batch_size, device=self.device),
        }

    def on_validation_epoch_end(self):
        """
        在验证 epoch 结束时聚合所有 batch 的指标。
        ✅ 确保 Lightning 记录的指标与实际测试结果一致
        """
        if not self.validation_outputs:
            return
        
        # 聚合所有 batch 的指标
        total_samples = sum(out["samples_evaluated"] for out in self.validation_outputs)
        total_batch_size = sum(out["batch_size"] for out in self.validation_outputs)
        
        # 加权平均（按样本数）
        if total_samples > 0:
            avg_seq_acc = sum(
                out["val_seq_acc"] * out["samples_evaluated"] 
                for out in self.validation_outputs
            ) / total_samples
            
            avg_validity = sum(
                out["val_validity"] * out["samples_evaluated"] 
                for out in self.validation_outputs
            ) / total_samples
            
            avg_similarity = sum(
                out["val_similarity"] * out["samples_evaluated"] 
                for out in self.validation_outputs
            ) / total_samples if any(out["val_similarity"] > 0 for out in self.validation_outputs) else 0.0
        else:
            avg_seq_acc = 0.0
            avg_validity = 0.0
            avg_similarity = 0.0
        
        # 记录聚合后的指标
        self.log("val_seq_acc_epoch", avg_seq_acc, prog_bar=True, sync_dist=False, logger=True)
        self.log("val_validity_epoch", avg_validity, prog_bar=True, sync_dist=False, logger=True)
        self.log("val_similarity_epoch", avg_similarity, prog_bar=True, sync_dist=False, logger=True)
        self.log("val_total_samples", float(total_samples), prog_bar=False, logger=True)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Validation Epoch {self.current_epoch} Summary:")
        logger.info(f"  Total Samples Evaluated: {total_samples} / {total_batch_size}")
        logger.info(f"  Sequence Accuracy: {avg_seq_acc:.4f}")
        logger.info(f"  Valid SMILES Ratio: {avg_validity:.4f}")
        logger.info(f"  Tanimoto Similarity: {avg_similarity:.4f}")
        logger.info(f"{'='*80}\n")
        
        # 清空输出列表
        self.validation_outputs = []
    
    def generate(self, c_peaks=None, h_peaks=None, h_features=None, formula_vector=None, 
                max_length=None, num_beams=1, do_sample=False, temperature=1.0, top_k=50, top_p=1.0, batch_size=None,
                c_nmr_mask=None, h_nmr_mask=None, **generate_kwargs):
        """
        Generate SMILES using T5 generation with optional formula guidance.
        Updated to support h_features for 1HNMR.
        """
        if max_length is None:
            max_length = self.config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
        
        # 2. 检查 Legacy Peak Input (根据配置)
        has_c = self.c_encoder is not None and c_peaks is not None
        has_h = self.h_encoder is not None and h_features is not None  # Use h_features instead of h_peaks
        has_nmr = has_c or has_h
        
        if not has_nmr:
            logger.error("Generate Check Failed: No NMR modality provided")
            raise ValueError(
                "至少需要提供一个NMR模态（C或H）："
                f"C-NMR (enabled={self.c_encoder is not None}, provided={c_peaks is not None}), "
                f"H-NMR (enabled={self.h_encoder is not None}, provided={h_features is not None})"
            )
        
        # ---------------------------------------------------------
        # Generation Logic
        # ---------------------------------------------------------
        # 确定设备
        device = next(self.parameters()).device
        
        # 处理单个样本情况（添加批次维度）
        # 优先从启用的NMR模态确定batch_size
        if batch_size is None:
            # 优先从C-NMR确定
            if c_peaks is not None:
                if c_peaks.dim() == 1:
                    c_peaks = c_peaks.unsqueeze(0)
                batch_size = c_peaks.shape[0]
            # 如果C-NMR不可用，从H-NMR确定
            elif h_features is not None:  # Use h_features instead of h_peaks
                if h_features.dim() == 1:
                    h_features = h_features.unsqueeze(0)
                batch_size = h_features.shape[0]
            # 如果NMR都不可用（不应该发生，因为前面已验证），从Formula确定
            elif formula_vector is not None:
                if formula_vector.dim() == 1:
                    formula_vector = formula_vector.unsqueeze(0)
                batch_size = formula_vector.shape[0]
            else:
                raise ValueError("无法确定batch_size：所有输入都是None")
        else:
            batch_size = batch_size # Explicitly set
        
        # 确保所有输入都在正确设备上
        if c_peaks is not None:
            c_peaks = c_peaks.to(device).float()
        if c_nmr_mask is not None:
            c_nmr_mask = c_nmr_mask.to(device).long()
        if h_features is not None:  # Use h_features instead of h_peaks
            h_features = h_features.to(device).float()
        if h_nmr_mask is not None:
            h_nmr_mask = h_nmr_mask.to(device).long()
        if formula_vector is not None:
            formula_vector = formula_vector.to(device).float()
        
        # 编码NMR数据（根据配置）
        z_c = None
        g_c = None
        if self.c_encoder is not None and c_peaks is not None:
            z_c, g_c, c_nmr_mask = self.c_encoder(c_peaks, mask=c_nmr_mask)
        
        z_h = None
        g_h = None
        if self.h_encoder is not None and h_features is not None:  # Use h_features
            z_h, g_h, h_nmr_mask = self.h_encoder(h_features, mask=h_nmr_mask)
        
        # 编码公式指导
        z_guidance = None
        if self.formula_encoder is not None and formula_vector is not None:
            # 确保formula_vector有正确的形状
            if formula_vector.dim() == 1:
                formula_vector = formula_vector.unsqueeze(0)
            # 验证向量维度
            expected_dim = self.config.FORMULA_VECTOR_SIZE
            actual_dim = formula_vector.shape[1]
            if actual_dim != expected_dim:
                pass # Warning already logged in validation?
            # For generation, we fix it silently or reuse logic
            if actual_dim < expected_dim:
                padding = torch.zeros(formula_vector.shape[0], expected_dim - actual_dim, device=device)
                formula_vector = torch.cat([formula_vector, padding], dim=1)
            else:
                formula_vector = formula_vector[:, :expected_dim]
            z_guidance = self.formula_encoder(formula_vector)
        
        # g_c = None
        # g_h = None
        # 融合所有模态
        encoder_hidden, attention_mask = self.fusion(z_c, z_h, g_c, g_h, z_guidance, c_nmr_mask, h_nmr_mask)
        
        # 创建encoder输出
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        try:
            # 使用T5生成
            generated_ids = self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.config.PAD_TOKEN_ID,
                eos_token_id=self.config.EOS_TOKEN_ID,
                bos_token_id=self.config.BOS_TOKEN_ID,
                **generate_kwargs,
            )
            return generated_ids
        except Exception as e:
            logger.error(f"T5 generation failed: {str(e)}")
            logger.error(f"encoder_outputs shape: {encoder_hidden.shape}")
            logger.error(f"attention_mask shape: {attention_mask.shape}")
            # 尝试更简单的生成方式作为回退
            try:
                logger.warning("Attempting fallback generation with minimal parameters")
                fallback_kwargs = dict(generate_kwargs)
                # 再次确保 beam 与返回序列数兼容
                fb_num_return_sequences = int(fallback_kwargs.get("num_return_sequences", 1))
                fb_num_beams = max(num_beams, fb_num_return_sequences)
                generated_ids = self.t5.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=fb_num_beams,
                    pad_token_id=self.config.PAD_TOKEN_ID,
                    eos_token_id=self.config.EOS_TOKEN_ID,
                    bos_token_id=self.config.BOS_TOKEN_ID,
                    **fallback_kwargs,
                )
                return generated_ids
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {str(e2)}")
                raise
    
    def tokens_to_smiles(self, tokens) -> str:
        """Convert token sequence to SMILES string."""
        if tokens is None or len(tokens) == 0:
            return ""
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        token_strings: List[str] = []
        found_eos = False
        
        for i, token in enumerate(tokens):
            # Stop at EOS token
            if found_eos:
                break
            
            token_id = int(token)
            
            # Handle special tokens - use them as markers but don't include in output
            if token_id == self.config.EOS_TOKEN_ID:
                found_eos = True
                break  # Stop here, don't include <eos> in output
            elif token_id == self.config.BOS_TOKEN_ID:
                continue  # Skip <bos>, don't include in output
            elif token_id == self.config.PAD_TOKEN_ID:
                continue  # Skip <pad>, don't include in output
            else:
                # Normal token
                try:
                    token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                    if isinstance(token_str, list):
                        token_str = token_str[0]
                    token_strings.append(token_str)
                except Exception:
                    # If tokenizer can't convert, skip unknown token
                    pass
        
        smiles = "".join(token_strings)
        smiles = re.sub(r"\s+", "", smiles)
        return smiles
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Use step-based warmup
        warmup_epochs = 5
        estimated_steps_per_epoch = 1000  # Conservative estimate
        
        def warmup_step(step):
            # Calculate warmup steps
            if hasattr(self, 'trainer') and self.trainer is not None:
                if hasattr(self.trainer, 'num_training_batches'):
                    steps_per_epoch = self.trainer.num_training_batches
                    if steps_per_epoch is not None and steps_per_epoch > 0:
                        warmup_steps = warmup_epochs * steps_per_epoch
                    else:
                        warmup_steps = warmup_epochs * estimated_steps_per_epoch
                else:
                    warmup_steps = warmup_epochs * estimated_steps_per_epoch
            else:
                warmup_steps = warmup_epochs * estimated_steps_per_epoch
            
            # Warmup: step 0 → 1/warmup_steps, step warmup_steps-1 → 1.0
            if step < warmup_steps:
                return min(1.0, (step + 1) / warmup_steps)
            return 1.0
        
        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, warmup_step),
            "interval": "step",  # Use step instead of epoch
        }
        
        return [optimizer], [scheduler]

    def _resize_t5_embeddings_to_tokenizer(self):
        """Resize T5 embeddings and LM head to match the custom SMILES tokenizer."""
        vocab_size = len(self.tokenizer)
        current_size = self.t5.config.vocab_size

        if vocab_size == current_size:
            return

        # 这一行会同时调整 shared embedding 和 lm_head 的权重尺寸
        self.t5.resize_token_embeddings(vocab_size)
        self.t5.config.vocab_size = vocab_size

__all__ = ["NMR2SMILESModel", "SetTransformerPeakEncoder", "SetTransformer1HNMRPeakEncoder", "FusionLayer"]
