"""
NMR2SMILES Model - Complete Implementation
改进版：多 Head 特征编码 + Fourier 连续值编码
"""

import logging
import re
import math
import json
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# ===========================================================
# 0. Fourier Features - 连续值编码核心
# ===========================================================
class FourierFeatures(nn.Module):
    """
    傅里叶特征编码，将连续标量映射到高维空间
    保留数值敏感性：接近的值→接近的向量表示
    """
    def __init__(self, x_min=0.0, x_max=1.0, num_freqs=64, 
                 funcs='both', strategy='lin_float_int', 
                 trainable=False, sigma=10):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.num_freqs = num_freqs
        self.funcs = funcs  # 'sin', 'cos', 'both'
        self.trainable = trainable
        
        # 计算频率
        if trainable:
            self.freqs = nn.Parameter(torch.randn(num_freqs) * sigma)
        else:
            # 固定频率：对数间隔，更好地覆盖不同尺度
            self.freqs = torch.logspace(0, math.log10(max(num_freqs, 2)), num_freqs)
        
        # 输出维度
        if funcs == 'both':
            self.out_dim = num_freqs * 2
        else:
            self.out_dim = num_freqs
    
    def forward(self, x):
        """
        x: (B, L, 1) 或 (B, L)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L) → (B, L, 1)
        
        # 归一化到 [0, 1]
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        x_norm = x_norm * 2 * math.pi  # [0, 2π]
        
        # 计算傅里叶特征
        freqs = self.freqs.to(x.device)  # (F,)
        x_freq = x_norm * freqs  # (B, L, F)
        
        features = []
        if self.funcs in ['sin', 'both']:
            features.append(torch.sin(x_freq))
        if self.funcs in ['cos', 'both']:
            features.append(torch.cos(x_freq))
        
        output = torch.cat(features, dim=-1)  # (B, L, out_dim)
        return output
    
    def num_features(self):
        return self.out_dim


# ===========================================================
# 1. Set Transformer Core Modules
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


# ===========================================================
# 2. C-NMR Peak Encoder (Simple)
# ===========================================================
class SetTransformerPeakEncoder(nn.Module):
    """
    Set Transformer Encoder for C-NMR peaks
    输入：(B, L, 2) [ppm, intensity]
    输出：(B, L, d_model) 峰特征编码
    """
    def __init__(self, d_model=512, n_heads=8, num_inds=32, num_seeds=32,
                 n_layers=3, ff_dim=1024, dropout=0.1, max_peaks=60):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        self.n_layers = n_layers
        
        # 1. 输入投影层：将 [ppm, intensity] 投影到 d_model 维度
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 2. ISAB 层（可配置层数）
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, d_model, n_heads, num_inds, dropout)
            for _ in range(n_layers)
        ])
        
        # 3. PMA 层（用于提取全局表示）
        self.pma = PMA(d_model, n_heads, num_seeds, dropout)
        
        # 4. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 5. 用于 padding 的可学习 token
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, peaks, mask=None):
        """
        peaks: (B, L, 2) - [ppm, intensity]
        mask:  (B, L)    - 1=valid, 0=padding
        """
        if peaks is None:
            return None, None, None
        
        if mask is None:
            mask = (peaks.abs().sum(-1) > 1e-6).long()
        
        # 1. 输入投影
        X = self.input_proj(peaks)  # (B, L, d_model)
        
        # 2. 应用 mask
        pad_mask = ~mask.bool()
        if pad_mask.any():
            pad_tokens = self.pad_token.expand(X.size(0), X.size(1), -1)
            X = torch.where(pad_mask.unsqueeze(-1), pad_tokens, X)
        
        # 3. Set Transformer 编码
        for isab_layer in self.isab_layers:
            X = isab_layer(X, mask=pad_mask if pad_mask.any() else None)
        
        # 4. PMA 全局表示
        global_repr = self.pma(X, mask=pad_mask if pad_mask.any() else None)
        
        # 5. 输出投影
        X = self.output_proj(X)
        
        return X, global_repr, mask.bool()


# ===========================================================
# 3. H-NMR Peak Encoder (Multi-Head Features) ⭐ 改进版
# ===========================================================
class SetTransformer1HNMRPeakEncoder(nn.Module):
    """
    多 Head 编码 1HNMR 特征，每类特征独立编码后融合
    输入：[chem_shift, peak_width, split_idx, integral, j1-j6] (10 维)
    
    改进点：
    1. chem_shift → Fourier 编码（保留数值敏感性）
    2. integral → Embedding（符合离散本质）
    3. J-coupling → Fourier×6 + 融合（保留内部关系）
    4. 特征融合 → Attention（学习特征权重）
    """
    def __init__(self, d_model=512, n_heads=8, num_inds=32, num_seeds=32,
                 n_layers=3, ff_dim=1024, dropout=0.1, max_peaks=60,
                 split_vocab_size=16, use_fourier=True):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        self.n_layers = n_layers
        self.split_vocab_size = split_vocab_size
        self.use_fourier = use_fourier
        
        # ========== 1. Shift Head (化学位移) ==========
        if use_fourier:
            self.shift_fourier = FourierFeatures(
                strategy='lin_float_int',
                x_min=0.0, x_max=12.0,  # H-NMR 范围
                num_freqs=64,
                funcs='both',  # sin + cos → 128 维
                trainable=False
            )
            self.shift_proj = nn.Sequential(
                nn.Linear(128, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model)
            )
        else:
            self.shift_proj = nn.Linear(1, d_model)
        
        # ========== 2. Width Head (峰宽) ==========
        self.width_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # ========== 3. Integral Head (积分/原子数) ==========
        self.max_integral = 20  # 最大原子数假设
        self.integral_embedding = nn.Embedding(self.max_integral + 1, d_model // 2)
        self.integral_proj = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # ========== 4. J-Coupling Head (耦合常数) ==========
        if use_fourier:
            self.j_fourier = FourierFeatures(
                x_min=0.0, x_max=20.0,  # J-coupling 范围 (Hz)
                num_freqs=16,
                funcs='both'  # → 64 维 per J
            )
            self.j_encoder = nn.Sequential(
                nn.Linear(32 * 6, d_model // 2),  # 6 个 J 值
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model)
            )
        else:
            self.j_encoder = nn.Linear(6, d_model)
        
        # ========== 5. Split Pattern Head (裂分模式) ==========
        self.split_embedding = nn.Embedding(split_vocab_size, d_model // 4)
        self.split_proj = nn.Sequential(
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # ========== 6. Feature Fusion (特征融合) ==========
        # Attention 融合 5 个 Head
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # ========== 7. Set Transformer (峰间关系) ==========
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, d_model, n_heads, num_inds, dropout)
            for _ in range(n_layers)
        ])
        self.pma = PMA(d_model, n_heads, num_seeds, dropout)
        
        # 8. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 9. Padding token
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def shuffle_peaks(self, peaks, mask=None):
        """仅在 mask=1 的有效位置内打乱 peaks"""
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
        peaks: (B, L, 10) - [chem_shift, width, split_idx, integral, j1-j6]
        mask:  (B, L)     - 1=valid, 0=padding
        """
        if peaks is None:
            return None, None, None
        
        if mask is None:
            mask = (peaks.abs().sum(-1) > 1e-6).long()
        
        batch_size, seq_len, _ = peaks.shape
        device = peaks.device
        
        # ========== 分离特征 ==========
        chem_shift = peaks[..., 0:1]      # (B, L, 1)
        peak_width = peaks[..., 1:2]      # (B, L, 1)
        split_idx = peaks[..., 2].long()  # (B, L)
        integral = peaks[..., 3:4]        # (B, L, 1)
        j_coupling = peaks[..., 4:10]     # (B, L, 6)
        
        # ========== 1. Shift Head ==========
        if self.use_fourier:
            shift_fourier = self.shift_fourier(chem_shift)  # (B, L, 128)
            shift_feat = self.shift_proj(shift_fourier)     # (B, L, d_model)
        else:
            shift_feat = self.shift_proj(chem_shift)
        
        # ========== 2. Width Head ==========
        width_feat = self.width_proj(peak_width)
        
        # ========== 3. Integral Head ==========
        integral_clamped = torch.clamp(integral.long(), 0, self.max_integral)
        integral_emb = self.integral_embedding(integral_clamped.squeeze(-1))
        integral_feat = self.integral_proj(integral_emb)
        
        # ========== 4. J-Coupling Head ==========
        if self.use_fourier:
            j_fourier_list = []
            for j_idx in range(6):
                j_val = j_coupling[..., j_idx:j_idx+1]
                j_fourier_list.append(self.j_fourier(j_val))
            j_fourier_cat = torch.cat(j_fourier_list, dim=-1)  # (B, L, 384)
            j_feat = self.j_encoder(j_fourier_cat)
        else:
            j_feat = self.j_encoder(j_coupling)
        
        # ========== 5. Split Head ==========
        split_emb = self.split_embedding(split_idx)
        split_feat = self.split_proj(split_emb)
        
        # ========== 6. Feature Fusion (Attention) ==========
        # 5 个 Head 作为 5 个"特征 token"
        head_features = torch.stack([
            shift_feat, width_feat, integral_feat, j_feat, split_feat
        ], dim=1)  # (B, 5, L, d_model)
        
        # 对每个位置 L，用 Attention 融合 5 个 Head
        head_features = head_features.permute(0, 2, 1, 3)  # (B, L, 5, d_model)
        head_features = head_features.reshape(batch_size * seq_len, 5, self.d_model)
        
        # Self-Attention 融合 5 个 Head
        fused, _ = self.fusion_attention(
            head_features, head_features, head_features
        )  # (B*L, 5, d_model)
        
        # 取第一个 token 作为融合结果
        fused = fused[:, 0, :]  # (B*L, d_model)
        fused = fused.view(batch_size, seq_len, self.d_model)
        fused = self.fusion_norm(fused)
        
        # ========== 7. 应用 Mask ==========
        pad_mask = ~mask.bool()
        if pad_mask.any():
            pad_tokens = self.pad_token.expand(batch_size, seq_len, -1)
            fused = torch.where(pad_mask.unsqueeze(-1), pad_tokens, fused)
        
        # ========== 8. Set Transformer ==========
        for isab_layer in self.isab_layers:
            fused = isab_layer(fused, mask=pad_mask if pad_mask.any() else None)
        
        # ========== 9. PMA 全局表示 ==========
        global_repr = self.pma(fused, mask=pad_mask if pad_mask.any() else None)
        
        # ========== 10. 输出投影 ==========
        output = self.output_proj(fused)
        
        return output, global_repr, mask.bool()


# ===========================================================
# 4. Multi-modal Fusion Layer
# ===========================================================
class FusionLayer(nn.Module):
    """
    融合层，合并 C-NMR + H-NMR + Formula Guidance
    返回特征和对应的 attention mask
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, z_c=None, z_h=None, global_c=None, global_h=None, 
                z_guidance=None, c_mask=None, h_mask=None):
        """
        Returns:
            z_all: (B, L_total, d_model) - 融合后的特征
            attention_mask: (B, L_total) - 对应的 attention mask
        """
        has_local = (z_c is not None) or (z_h is not None)
        has_global = (global_c is not None) or (global_h is not None)
        has_guidance = z_guidance is not None
        
        if not (has_local or has_global or has_guidance):
            raise ValueError("Need at least one input feature")
        
        all_features = []
        all_masks = []
        
        # 确定 batch_size 和 device
        batch_size = None
        device = None
        for tensor in [z_c, z_h, global_c, global_h, z_guidance]:
            if tensor is not None:
                batch_size = tensor.size(0)
                device = tensor.device
                break
        
        # 1. 局部特征
        if z_c is not None:
            all_features.append(z_c)
            all_masks.append(c_mask if c_mask is not None else 
                           torch.ones(z_c.shape[:2], dtype=torch.long, device=device))
        
        if z_h is not None:
            all_features.append(z_h)
            all_masks.append(h_mask if h_mask is not None else 
                           torch.ones(z_h.shape[:2], dtype=torch.long, device=device))
        
        # 2. 全局特征
        if global_c is not None:
            all_features.append(global_c)
            all_masks.append(torch.ones(global_c.shape[:2], dtype=torch.long, device=device))
        
        if global_h is not None:
            all_features.append(global_h)
            all_masks.append(torch.ones(global_h.shape[:2], dtype=torch.long, device=device))
        
        # 3. Formula Guidance
        if z_guidance is not None:
            all_features.append(z_guidance)
            all_masks.append(torch.ones(z_guidance.shape[:2], dtype=torch.long, device=device))
        
        # 4. 拼接
        z_all = torch.cat(all_features, dim=1)
        attention_mask = torch.cat(all_masks, dim=1)
        
        return z_all, attention_mask


# ===========================================================
# 5. Formula Encoder
# ===========================================================
class FormulaEncoder(nn.Module):
    """将化学式向量转换为 guidance embedding"""
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
        x = self.net(formula_vector)
        return x.unsqueeze(1)  # (B, 1, d_model)


# ===========================================================
# 6. Main Model: NMR2SMILESModel
# ===========================================================
class NMR2SMILESModel(pl.LightningModule):
    """
    主模型：NMR → SMILES 生成
    架构：SetTransformer Encoder + T5 Decoder
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        d_model = config.PEAK_ENCODER_D_MODEL
        
        # 消融实验配置
        use_c_nmr = getattr(config, "USE_C_NMR", True)
        use_h_nmr = getattr(config, "USE_H_NMR", True)
        use_formula = getattr(config, "USE_FORMULA_GUIDANCE", True)
        
        if not (use_c_nmr or use_h_nmr):
            raise ValueError("至少需要启用一个 NMR 模态")
        
        logger.info("=" * 80)
        logger.info("消融实验配置:")
        logger.info(f"  USE_C_NMR: {use_c_nmr}")
        logger.info(f"  USE_H_NMR: {use_h_nmr}")
        logger.info(f"  USE_FORMULA_GUIDANCE: {use_formula}")
        logger.info("=" * 80)
        
        # Split 词汇表
        self.split_vocab = [
            '<unk>', 'm', 'd', 's', 'dd', 't', 'ddd', 'q',
            'dt', 'td', 'br', 'ddt', 'dq', 'tt', 'quint',
            'dddd', 'qd', 'sept', 'ddp', 'ddq', 'bd', 'dqd',
        ]
        self.split_to_idx = {token: idx for idx, token in enumerate(self.split_vocab)}
        
        # C-NMR Encoder
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
            logger.info("❌ C-NMR encoder 已禁用")
        
        # H-NMR Encoder (Multi-Head)
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
                split_vocab_size=len(self.split_vocab),
                use_fourier=getattr(config, "USE_FOURIER_ENCODING", True)
            )
            logger.info("✅ H-NMR encoder (Multi-Head) 已启用")
        else:
            self.h_encoder = None
            logger.info("❌ H-NMR encoder 已禁用")
        
        # Formula Encoder
        if use_formula:
            if not hasattr(config, "ALL_ATOMS"):
                raise ValueError("config.ALL_ATOMS must be set")
            self.formula_encoder = FormulaEncoder(
                formula_vector_size=len(config.ALL_ATOMS),
                d_model=d_model,
            )
            logger.info(f"✅ Formula encoder initialized with {len(config.ALL_ATOMS)} atoms")
        else:
            self.formula_encoder = None
        
        # Fusion Layer
        self.fusion = FusionLayer(d_model=d_model)
        
        # T5 Decoder
        logger.debug(f"Initializing T5 model from: {config.T5_MODEL_NAME}")
        if getattr(config, "USE_RANDOM_T5_INIT", False):
            t5_config = T5Config.from_pretrained(config.T5_MODEL_NAME, local_files_only=True)
            t5_config.vocab_size = len(tokenizer)
            self.t5 = T5ForConditionalGeneration(t5_config)
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                config.T5_MODEL_NAME,
                local_files_only=True
            )
        
        self._resize_t5_embeddings_to_tokenizer()
        
        # 移除 Cross-Attention 位置偏置（集合编码不需要）
        if getattr(config, "REMOVE_CROSS_ATTENTION_POSITION_BIAS", False):
            for layer in self.t5.decoder.block:
                cross_attention = layer.layer[1].EncDecAttention
                cross_attention.has_relative_attention_bias = False
                cross_attention.relative_attention_bias = None
        
        # 可选：冻结 T5 Decoder
        if config.FREEZE_T5_DECODER:
            logger.info("🔒 Freezing T5 decoder parameters")
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
        
        self.t5.train()
        self.validation_outputs = []
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
                chem_shift = float(peak[0])
                split_str = str(peak[2]).strip().lower() if len(peak) > 2 else "<unk>"
                split_idx = self.split_to_idx.get(split_str, self.split_to_idx["<unk>"])
                integral_str = str(peak[3]).strip() if len(peak) > 3 else '1H'
                integral_value = 1.0
                match = re.search(r'(\d+)(?:H|h)?', integral_str)
                if match:
                    integral_value = float(match.group(1))
                j_coupling = peak[4] if len(peak) > 4 and isinstance(peak[4], list) else []
                padded_j = self._pad_j_coupling(j_coupling)
                peak_features = [chem_shift, 0.0, split_idx, integral_value] + padded_j
                features.append(peak_features)
            return features
        except Exception as e:
            logger.warning(f"Error preparing H-NMR features: {str(e)}")
            return []
    
    def forward(self, c_peaks=None, h_peaks=None, formula=None, smiles_ids=None,
                attention_mask=None, formula_vector=None, c_nmr_mask=None, 
                h_nmr_mask=None, h_features=None, **kwargs):
        """
        输入:
            c_peaks: (B, L, 2) - C-NMR [ppm, intensity]
            h_features: (B, L, 10) - H-NMR 多特征
            smiles_ids: tokenized smiles (labels)
        """
        # C-NMR
        z_c, global_c, c_nmr_mask = None, None, None
        if self.c_encoder is not None and c_peaks is not None:
            z_c, global_c, c_nmr_mask = self.c_encoder(c_peaks, mask=c_nmr_mask)
        
        # H-NMR
        z_h, global_h, h_nmr_mask = None, None, None
        if self.h_encoder is not None and h_features is not None:
            z_h, global_h, h_nmr_mask = self.h_encoder(h_features, mask=h_nmr_mask)
        
        # Formula Guidance
        z_guidance = None
        if self.formula_encoder is not None and formula_vector is not None:
            z_guidance = self.formula_encoder(formula_vector)
        
        # Fusion
        encoder_hidden, attention_mask = self.fusion(
            z_c, z_h, global_c, global_h, z_guidance,
            c_mask=c_nmr_mask, h_mask=h_nmr_mask
        )
        
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        # T5 Decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=smiles_ids,
        )
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        smiles_ids = batch["smiles"].long()
        c_peaks = batch.get("c_nmr_peaks") if self.c_encoder is not None else None
        c_nmr_mask = batch.get("c_nmr_mask") if self.c_encoder is not None else None
        h_features = batch.get("h_nmr_features") if self.h_encoder is not None else None
        h_nmr_mask = batch.get("h_nmr_mask") if self.h_encoder is not None else None
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None
        
        # Check NaN
        if c_peaks is not None and torch.isnan(c_peaks).any():
            logger.warning(f"NaN detected in c_peaks at batch {batch_idx}")
        if h_features is not None and torch.isnan(h_features).any():
            logger.warning(f"NaN detected in h_features at batch {batch_idx}")
        
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        outputs = self(
            c_peaks=c_peaks,
            h_features=h_features,
            formula_vector=formula_vector,
            smiles_ids=labels,
            c_nmr_mask=c_nmr_mask,
            h_nmr_mask=h_nmr_mask,
        )
        
        loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf loss at batch {batch_idx}")
            return None
        
        # Metrics
        with torch.no_grad():
            logits = outputs.logits
            pred_tokens = logits.argmax(dim=-1)
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float()
            seq_correct = ((pred_tokens == smiles_ids) | ~valid_mask).all(dim=1)
            seq_acc = seq_correct.float().mean()
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, logger=True)
        self.log("train_token_acc", token_acc, prog_bar=True, sync_dist=True, logger=True)
        self.log("train_seq_acc", seq_acc, prog_bar=True, sync_dist=True, logger=True)
        
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=False, logger=True)
        
        return loss
    
    @staticmethod
    def evaluate_smiles_pair(pred_smiles, origin_smiles):
        """Evaluate a single pair of SMILES using RDKit"""
        acc, valid, similarity = 0.0, 0.0, 0.0
        pred_mol = None
        try:
            pred_mol = Chem.MolFromSmiles(pred_smiles)
        except Exception:
            pass
        
        if pred_mol is None:
            return acc, valid, similarity
        
        valid = 1.0
        try:
            origin_mol = Chem.MolFromSmiles(origin_smiles)
            if origin_mol is None:
                return acc, valid, similarity
            if Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(origin_mol):
                acc = 1.0
            fp_1 = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048, useChirality=True)
            fp_2 = AllChem.GetMorganFingerprintAsBitVect(origin_mol, 2, nBits=2048, useChirality=True)
            similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
        except Exception:
            pass
        
        return acc, valid, similarity
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.global_rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting validation at epoch {self.current_epoch}")
            logger.info(f"{'='*80}")
        
        smiles_ids = batch["smiles"].long()
        c_peaks = batch.get("c_nmr_peaks") if self.c_encoder is not None else None
        c_nmr_mask = batch.get("c_nmr_mask") if self.c_encoder is not None else None
        h_features = batch.get("h_nmr_features") if self.h_encoder is not None else None
        h_nmr_mask = batch.get("h_nmr_mask") if self.h_encoder is not None else None
        original_smiles_list = batch["original_smiles"]
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None
        
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
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float()
            self.log("val_token_acc", token_acc, prog_bar=True, sync_dist=True, logger=True)
        
        # RDKit Evaluation
        val_mol_acc = torch.tensor(0.0, device=self.device)
        val_validity = torch.tensor(0.0, device=self.device)
        val_similarity = torch.tensor(0.0, device=self.device)
        seq_acc = torch.tensor(0.0, device=self.device)
        
        try:
            gen_c_peaks = c_peaks[:1] if c_peaks is not None else None
            gen_c_mask = c_nmr_mask[:1] if c_nmr_mask is not None else None
            gen_h_features = h_features[:1] if h_features is not None else None
            gen_h_mask = h_nmr_mask[:1] if h_nmr_mask is not None else None
            gen_formula = formula_vector[:1] if formula_vector is not None else None
            
            generated_ids = self.generate(
                c_peaks=gen_c_peaks,
                h_features=gen_h_features,
                formula_vector=gen_formula,
                c_nmr_mask=gen_c_mask,
                h_nmr_mask=gen_h_mask,
                max_length=self.config.MAX_SMILES_LENGTH + 5,
                num_beams=1,
                do_sample=False
            )
            
            acc_list, valid_list, sim_list = [], [], []
            rdBase.DisableLog('rdApp.error')
            try:
                for i in range(len(generated_ids)):
                    if i >= len(original_smiles_list):
                        break
                    pred_smiles = self.tokens_to_smiles(generated_ids[i])
                    true_smiles = original_smiles_list[i]
                    acc, valid, sim = self.evaluate_smiles_pair(pred_smiles, true_smiles)
                    acc_list.append(acc)
                    valid_list.append(valid)
                    sim_list.append(sim)
            finally:
                rdBase.EnableLog('rdApp.error')
            
            if len(valid_list) > 0:
                val_mol_acc = torch.tensor(acc_list, device=self.device).float().mean()
                val_validity = torch.tensor(valid_list, device=self.device).float().mean()
                val_similarity = torch.tensor(sim_list, device=self.device).float().mean()
                seq_acc = val_mol_acc
        except Exception as e:
            logger.warning(f"Validation generation failed: {e}")
        
        self.log("val_seq_acc", seq_acc, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_validity", val_validity, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_similarity", val_similarity, prog_bar=True, sync_dist=True, logger=True)
        
        # Log examples
        if batch_idx % 10 == 0 and self.global_rank == 0:
            try:
                generated = self.generate(
                    c_peaks=c_peaks[:1] if c_peaks is not None else None,
                    h_features=h_features[:1] if h_features is not None else None,
                    formula_vector=formula_vector[:1] if formula_vector is not None else None,
                    c_nmr_mask=c_nmr_mask[:1] if c_nmr_mask is not None else None,
                    h_nmr_mask=h_nmr_mask[:1] if h_nmr_mask is not None else None,
                )
                generated_smiles = self.tokens_to_smiles(generated[0])
                original_smiles = original_smiles_list[0]
                logger.info(f"\n[Validation Example {batch_idx}]")
                logger.info(f"Original:  {original_smiles}")
                logger.info(f"Generated: {generated_smiles}")
            except Exception as e:
                logger.warning(f"Failed to log example: {e}")
        
        return {"val_token_acc": token_acc, "val_seq_acc": seq_acc}
    
    def generate(self, c_peaks=None, h_peaks=None, h_features=None, formula_vector=None,
                 max_length=None, num_beams=1, do_sample=False, temperature=1.0,
                 top_k=50, top_p=1.0, batch_size=None, c_nmr_mask=None, 
                 h_nmr_mask=None, **generate_kwargs):
        """Generate SMILES using T5 generation"""
        if max_length is None:
            max_length = self.config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
        
        has_c = self.c_encoder is not None and c_peaks is not None
        has_h = self.h_encoder is not None and h_features is not None
        
        if not (has_c or has_h):
            raise ValueError("至少需要提供一个 NMR 模态（C 或 H）")
        
        device = next(self.parameters()).device
        
        # Determine batch_size
        if batch_size is None:
            if c_peaks is not None:
                if c_peaks.dim() == 1:
                    c_peaks = c_peaks.unsqueeze(0)
                batch_size = c_peaks.shape[0]
            elif h_features is not None:
                if h_features.dim() == 1:
                    h_features = h_features.unsqueeze(0)
                batch_size = h_features.shape[0]
            elif formula_vector is not None:
                if formula_vector.dim() == 1:
                    formula_vector = formula_vector.unsqueeze(0)
                batch_size = formula_vector.shape[0]
            else:
                raise ValueError("无法确定 batch_size")
        
        # Move to device
        if c_peaks is not None:
            c_peaks = c_peaks.to(device).float()
        if c_nmr_mask is not None:
            c_nmr_mask = c_nmr_mask.to(device).long()
        if h_features is not None:
            h_features = h_features.to(device).float()
        if h_nmr_mask is not None:
            h_nmr_mask = h_nmr_mask.to(device).long()
        if formula_vector is not None:
            formula_vector = formula_vector.to(device).float()
        
        # Encode
        z_c, g_c = None, None
        if self.c_encoder is not None and c_peaks is not None:
            z_c, g_c, c_nmr_mask = self.c_encoder(c_peaks, mask=c_nmr_mask)
        
        z_h, g_h = None, None
        if self.h_encoder is not None and h_features is not None:
            z_h, g_h, h_nmr_mask = self.h_encoder(h_features, mask=h_nmr_mask)
        
        z_guidance = None
        if self.formula_encoder is not None and formula_vector is not None:
            if formula_vector.dim() == 1:
                formula_vector = formula_vector.unsqueeze(0)
            z_guidance = self.formula_encoder(formula_vector)
        
        # Fusion
        encoder_hidden, attention_mask = self.fusion(
            z_c, z_h, g_c, g_h, z_guidance, c_nmr_mask, h_nmr_mask
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        # Generate
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
    
    def tokens_to_smiles(self, tokens) -> str:
        """Convert token sequence to SMILES string"""
        if tokens is None or len(tokens) == 0:
            return ""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        token_strings = []
        found_eos = False
        
        for token in tokens:
            if found_eos:
                break
            token_id = int(token)
            
            if token_id == self.config.EOS_TOKEN_ID:
                found_eos = True
                break
            elif token_id == self.config.BOS_TOKEN_ID:
                continue
            elif token_id == self.config.PAD_TOKEN_ID:
                continue
            else:
                try:
                    token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                    if isinstance(token_str, list):
                        token_str = token_str[0]
                    token_strings.append(token_str)
                except Exception:
                    pass
        
        smiles = "".join(token_strings)
        smiles = re.sub(r"\s+", "", smiles)
        return smiles
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        warmup_epochs = 5
        estimated_steps_per_epoch = 1000
        
        def warmup_step(step):
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
            
            if step < warmup_steps:
                return min(1.0, (step + 1) / warmup_steps)
            return 1.0
        
        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, warmup_step),
            "interval": "step",
        }
        
        return [optimizer], [scheduler]
    
    def _resize_t5_embeddings_to_tokenizer(self):
        """Resize T5 embeddings and LM head to match custom tokenizer"""
        vocab_size = len(self.tokenizer)
        current_size = self.t5.config.vocab_size
        
        if vocab_size == current_size:
            return
        
        self.t5.resize_token_embeddings(vocab_size)
        self.t5.config.vocab_size = vocab_size


__all__ = [
    "NMR2SMILESModel",
    "SetTransformerPeakEncoder",
    "SetTransformer1HNMRPeakEncoder",
    "FusionLayer",
    "FormulaEncoder",
    "FourierFeatures",
]