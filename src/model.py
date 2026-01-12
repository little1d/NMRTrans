import logging
import re
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem



# ===========================================================
# 1. PeakEncoder —— 用于 H-NMR 与 C-NMR（输入 list[float]）
# ===========================================================

class PeakEncoder(nn.Module):
    """
    输入: (B, L) 的 ppm list → 输出: (B, L, d_model) 的峰序列编码
    """
    def __init__(self, d_model=512, n_layers=6, n_heads=8, ff_dim=2048, dropout=0.1, max_peaks=60):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        
        # 1. 峰值投影
        self.input_proj = nn.Linear(1, d_model)
        
        # 2. 峰顺序嵌入（learnable positional embedding）
        self.peak_order_embedding = nn.Embedding(max_peaks, d_model)
        
        # 3. 输入 LayerNorm
        self.input_norm = nn.LayerNorm(d_model)
        
        # 4. Transformer 编码器（带 dropout，使用 Pre-LN 提升梯度稳定性）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=ff_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: 在残差相加前进行归一化，梯度更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. 输出 LayerNorm
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, peaks, mask=None):
        """
        peaks: list of lists → 已转换成 padded tensor (B, L) or (B, L, 1)
        mask: (B, L) with 1 for valid peaks and 0 for padding (optional)
        returns: (B, L, d_model) - 保留序列维度
        """
        if peaks is None:
            return None

        # Handle both (B, L) and (B, L, 1) shapes
        if peaks.dim() == 2:
            # (B, L) -> (B, L, 1)
            x = peaks.unsqueeze(-1).float()
        elif peaks.dim() == 3:
            # (B, L, 1) -> keep as is
            x = peaks.float()
        else:
            raise ValueError(f"Expected peaks to have 2 or 3 dimensions, got {peaks.dim()}")

        batch_size, num_peaks, _ = x.shape
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf detected in peaks input!")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. 峰值投影
        x = self.input_proj(x)    # (B, L, d_model)

        # 2. 添加峰顺序嵌入（类似位置编码）
        peak_indices = torch.arange(num_peaks, device=x.device).unsqueeze(0).expand(batch_size, -1)
        peak_indices = torch.clamp(peak_indices, max=self.max_peaks - 1)  # 防止超出范围
        order_emb = self.peak_order_embedding(peak_indices)  # (B, L, d_model)
        x = x + order_emb
        
        # 3. 输入归一化
        x = self.input_norm(x)

        # 4. Transformer 编码（带 dropout）
        if mask is not None:
            # src_key_padding_mask: True for padding positions, False for valid
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)   # (B, L, d_model)

        # Check for NaN/Inf after transformer
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf detected after transformer encoding!")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. 输出归一化
        x = self.output_norm(x)
        
        return x   # (B, L, d_model)


# ===========================================================
# 2. Multi-modal Fusion Layer —— 合并 CNMR + HNMR + guidance
# ===========================================================

class FusionLayer(nn.Module):
    """
    融合:
    - z_c (C-NMR embedding)
    - z_h (H-NMR embedding)
    - z_guidance (Formula/MS/Scaffold) 未来可插拔

    当前实现: z_nmr = mean(C, H)
    guidance 作为 bias 加到输出 embedding 上
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model

    def forward(self, z_c, z_h, z_guidance=None):
        # 主模态 C/H 融合
        # z_c: (B, L_c, d_model)
        # z_h: (B, L_h, d_model)
        nmr_list = []

        if z_c is not None:
            nmr_list.append(z_c)
        if z_h is not None:
            nmr_list.append(z_h)

        if len(nmr_list) == 0:
            raise ValueError("No NMR modality provided! Need at least C or H.")

        # ✅ 在序列维度拼接，保留所有峰点信息
        # 而不是压缩成一个向量
        z_nmr = torch.cat(nmr_list, dim=1)   # (B, L_c + L_h, d_model)

        # 指导模态（formula/MS等）作为 bias
        if z_guidance is not None:
            # 如果有 guidance，在序列维度拼接
            z_all = torch.cat([z_nmr, z_guidance], dim=1)
        else:
            z_all = z_nmr

        # 返回完整的序列，让 T5 decoder 可以 attend 到所有峰点
        return z_all   # (B, L_total, d_model)

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
        
        # 主模态 encoder（根据配置初始化）
        if use_c_nmr:
            self.c_encoder = PeakEncoder(
                d_model=d_model,
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
            self.h_encoder = PeakEncoder(
                d_model=d_model,
                n_layers=config.PEAK_ENCODER_N_LAYERS,
                n_heads=config.PEAK_ENCODER_N_HEADS,
                ff_dim=config.PEAK_ENCODER_FF_DIM,
                dropout=config.PEAK_ENCODER_DROPOUT,
                max_peaks=config.MAX_PEAKS
            )
            logger.info("✅ H-NMR encoder 已启用")
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

    def forward(
        self,
        c_peaks=None,
        h_peaks=None,
        formula=None,
        smiles_ids=None,
        attention_mask=None,
        input_ids=None,  # Added for tokenized spectra
        formula_vector=None,
        c_nmr_mask=None,
        h_nmr_mask=None,
        **kwargs
    ):
        """
        输入:
          c_peaks: List[List[float]] → 已 padded tensor
          h_peaks: 同上
          smiles_ids: tokenized smiles (labels)
          input_ids: Tokenized spectra (for NMRMind mode)
        """
        
        # 1. 如果提供了 input_ids，直接使用 T5 encoder（NMRMind 模式）
        if input_ids is not None:
            # T5 encoder 接受 input_ids 和 attention_mask
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # encoder_outputs 是 BaseModelOutput
        else:
            # 2. 否则使用 PeakEncoder + Fusion（原始模式）
            
            # C-NMR（根据配置和输入决定）
            z_c = None
            if self.c_encoder is not None and c_peaks is not None:
                z_c = self.c_encoder(c_peaks, mask=c_nmr_mask)

            # H-NMR（根据配置和输入决定）
            z_h = None
            if self.h_encoder is not None and h_peaks is not None:
                z_h = self.h_encoder(h_peaks, mask=h_nmr_mask)

            # 新增：处理化学式guidance
            z_guidance = None
            if self.formula_encoder is not None and formula_vector is not None:
                z_guidance = self.formula_encoder(formula_vector)  # (B, 1, d_model)
            
            # 融合
            encoder_hidden = self.fusion(z_c, z_h, z_guidance)  # (B, L_total, d_model)

            # 构造 encoder_outputs（跳过 T5 encoder）
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

            # Create encoder attention mask for the peak sequence
            # encoder_hidden: (B, L, d_model) where L = num_c_peaks + num_h_peaks (+ formula)
            batch_size, seq_len, _ = encoder_hidden.shape

            mask_list = []
            if c_nmr_mask is not None:
                mask_list.append(c_nmr_mask)
            if h_nmr_mask is not None:
                mask_list.append(h_nmr_mask)

            if len(mask_list) > 0:
                encoder_attention_mask = torch.cat(mask_list, dim=1)
                if z_guidance is not None:
                    formula_mask = torch.ones(batch_size, 1, dtype=torch.long, device=encoder_attention_mask.device)
                    encoder_attention_mask = torch.cat([encoder_attention_mask, formula_mask], dim=1)
            else:
                # Fallback: no masks provided
                encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=encoder_hidden.device)

            if encoder_attention_mask.shape[1] != seq_len:
                if encoder_attention_mask.shape[1] > seq_len:
                    encoder_attention_mask = encoder_attention_mask[:, :seq_len]
                else:
                    padding = torch.zeros(
                        batch_size, seq_len - encoder_attention_mask.shape[1],
                        dtype=torch.long, device=encoder_attention_mask.device
                    )
                    encoder_attention_mask = torch.cat([encoder_attention_mask, padding], dim=1)
            
            # Use the constructed mask
            attention_mask = encoder_attention_mask

        # 送入 T5 decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,  # attention mask for encoder outputs
            labels=smiles_ids,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step for autoregressive generation."""
        smiles_ids = batch["smiles"].long()
        
        # 根据配置获取输入（消融实验）
        c_peaks = batch.get("c_nmr_peaks") if self.c_encoder is not None else None
        h_peaks = batch.get("h_nmr_peaks") if self.h_encoder is not None else None
        c_nmr_mask = batch.get("c_nmr_mask") if self.c_encoder is not None else None
        h_nmr_mask = batch.get("h_nmr_mask") if self.h_encoder is not None else None
        
        # Check for NaN in input data
        if c_peaks is not None and torch.isnan(c_peaks).any():
            logger.warning(f"NaN detected in c_peaks at batch {batch_idx}")
        if h_peaks is not None and torch.isnan(h_peaks).any():
            logger.warning(f"NaN detected in h_peaks at batch {batch_idx}")
        
        # T5 expects labels with -100 for positions to ignore
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None

        input_ids = batch.get("input_ids") # Check for input_ids
        attention_mask = batch.get("attention_mask")

        # Forward pass
        outputs = self(
            c_peaks=c_peaks,
            h_peaks=h_peaks,
            formula_vector=formula_vector,
            smiles_ids=labels,
            c_nmr_mask=c_nmr_mask,
            h_nmr_mask=h_nmr_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        loss = outputs.loss
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf loss detected at batch {batch_idx}")
            logger.error(f"C peaks shape: {c_peaks.shape if c_peaks is not None else None}")
            logger.error(f"H peaks shape: {h_peaks.shape if h_peaks is not None else None}")
            logger.error(f"SMILES ids shape: {smiles_ids.shape}")
            logger.error(f"Labels shape: {labels.shape}")
            
            # Check data range
            if c_peaks is not None:
                logger.error(f"C peaks range: [{c_peaks.min():.4f}, {c_peaks.max():.4f}]")
            if h_peaks is not None:
                logger.error(f"H peaks range: [{h_peaks.min():.4f}, {h_peaks.max():.4f}]")
            
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
            fp_1 = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048, useChirality=True)
            fp_2 = AllChem.GetMorganFingerprintAsBitVect(origin_mol, 2, nBits=2048, useChirality=True)
            similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
            
        except Exception as e:
            # logger.warning(f"RDKit evaluation error: {e}")
            pass
            
        return acc, valid, similarity

    def validation_step(self, batch, batch_idx):
        """Validation step with autoregressive generation and RDKit evaluation."""
        if batch_idx == 0 and self.global_rank == 0:
            logger.info(f"\\n{'='*80}")
            logger.info(f"Starting validation at epoch {self.current_epoch}")
            logger.info(f"{'='*80}")
        
        smiles_ids = batch["smiles"].long()
        
        # 根据配置获取输入（消融实验）
        c_peaks = batch.get("c_nmr_peaks") if self.c_encoder is not None else None
        h_peaks = batch.get("h_nmr_peaks") if self.h_encoder is not None else None
        c_nmr_mask = batch.get("c_nmr_mask") if self.c_encoder is not None else None
        h_nmr_mask = batch.get("h_nmr_mask") if self.h_encoder is not None else None
        original_smiles_list = batch["original_smiles"]
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None

        # Teacher forcing evaluation
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        with torch.no_grad():
            outputs = self(
                c_peaks=c_peaks,
                h_peaks=h_peaks,
                formula_vector=formula_vector,
                smiles_ids=labels,
                c_nmr_mask=c_nmr_mask,
                h_nmr_mask=h_nmr_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            pred_tokens = logits.argmax(dim=-1)
            
            # Token accuracy
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float()
            
            # Token accuracy
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float()
        
        self.log("val_token_acc", token_acc, prog_bar=True, sync_dist=True, logger=True)

        # Generation & Evaluation with RDKit (BATCH)
        val_mol_acc = torch.tensor(0.0, device=self.device)
        val_validity = torch.tensor(0.0, device=self.device)
        val_similarity = torch.tensor(0.0, device=self.device)
        
        # Default seq_acc to 0.0 (if generation fails completely)
        seq_acc = torch.tensor(0.0, device=self.device)

        try:
            generated_ids = self.generate(
                c_peaks, h_peaks, formula_vector, 
                c_nmr_mask=c_nmr_mask, h_nmr_mask=h_nmr_mask,
                input_ids=input_ids, attention_mask=attention_mask, # Pass tokenized inputs
                max_length=self.config.MAX_SMILES_LENGTH + 5,
                num_beams=1, # Greedy decoding for validation speed
                do_sample=False
            )
            
            acc_list = []
            valid_list = []
            sim_list = []
            
            # Suppress RDKit error logs during evaluation to avoid flooding console with invalid SMILES errors
            rdBase.DisableLog('rdApp.error') 
            try:
                for i in range(len(generated_ids)):
                    if i >= len(original_smiles_list): break
                    
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
                
                # Update seq_acc to use RDKit exact match results
                seq_acc = val_mol_acc
                
        except Exception as e:
            logger.warning(f"Validation generation/evaluation failed: {e}")
        
        # Log metrics safely (Always logged)
        self.log("val_seq_acc", seq_acc, prog_bar=True, sync_dist=True, logger=True) # Purely RDKit-based
        self.log("val_validity", val_validity, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_similarity", val_similarity, prog_bar=True, sync_dist=True, logger=True)
        
        # Generate and log examples
        if batch_idx % 10 == 0 and self.global_rank == 0:
            try:
                # 根据配置安全地获取单个样本用于生成
                gen_c_peaks = c_peaks[:1] if c_peaks is not None else None
                gen_h_peaks = h_peaks[:1] if h_peaks is not None else None
                gen_formula = formula_vector[:1] if formula_vector is not None else None
                gen_c_mask = c_nmr_mask[:1] if c_nmr_mask is not None else None
                gen_h_mask = h_nmr_mask[:1] if h_nmr_mask is not None else None
                
                gen_input_ids = input_ids[:1] if input_ids is not None else None
                gen_attention_mask = attention_mask[:1] if attention_mask is not None else None

                generated = self.generate(
                    gen_c_peaks,
                    gen_h_peaks,
                    gen_formula,
                    c_nmr_mask=gen_c_mask,
                    h_nmr_mask=gen_h_mask,
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                )
                generated_smiles = self.tokens_to_smiles(generated[0])
                original_smiles = original_smiles_list[0]
                
                logger.info(f"\n[Validation Example {batch_idx}]")
                logger.info(f"Original:  {original_smiles}")
                logger.info(f"Generated: {generated_smiles}")
                
                self.validation_outputs.append({
                    "original": original_smiles,
                    "predicted_original": generated_smiles,
                    "val_token_acc": token_acc.item(),
                    "val_seq_acc": seq_acc.item(),
                })
            except Exception as e:
                logger.warning(f"Failed to generate validation example: {e}")
        
        return {"val_token_acc": token_acc, "val_seq_acc": seq_acc}
    
    def generate(self, c_peaks=None, h_peaks=None, formula_vector=None, input_ids=None, attention_mask=None, max_length=None, num_beams=1, 
                do_sample=False, temperature=1.0, top_k=50, top_p=1.0, batch_size=None,
                c_nmr_mask=None, h_nmr_mask=None, **generate_kwargs):
        """
        Generate SMILES using T5 generation with optional formula guidance.
        
        Args:
            c_peaks: C-NMR peaks tensor of shape (batch_size, num_peaks) or (num_peaks,)
            h_peaks: H-NMR peaks tensor of shape (batch_size, num_peaks) or (num_peaks,)
            formula_vector: Formula vector tensor of shape (batch_size, formula_vector_size) or (formula_vector_size,)
            max_length: Maximum length of generated SMILES
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            batch_size: Explicit batch size (optional)
        
        Returns:
            generated_ids: Generated token IDs tensor of shape (batch_size, sequence_length)
        """
        if max_length is None:
            max_length = self.config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
            
        # 1. 优先处理 Tokenized Input (NMRMind)
        if input_ids is not None:
            # Use T5 encoder directly
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            # Ensure input_ids on device
            input_ids = input_ids.to(self.device if hasattr(self, 'device') else input_ids.device)
            attention_mask = attention_mask.to(input_ids.device)
            
            # Encoder forward
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Prepare for generation (Batch size etc)
            batch_size_actual = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # Skip PeakEncoder logic and go directly to generation
            # We need to construct mask_list or just use attention_mask for decoder
            # In T5 generation, attention_mask is used for encoder_outputs usually.
            
            # Do Common Generation
            try:
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
                logger.error(f"T5 generation with input_ids failed: {str(e)}")
                raise e
        
        # 2. 检查 Legacy Peak Input (根据配置)
        has_c = self.c_encoder is not None and c_peaks is not None
        has_h = self.h_encoder is not None and h_peaks is not None
        has_nmr = has_c or has_h
        
        if not has_nmr:
            logger.error("Legacy Generate Check Failed: No NMR modality provided")
            raise ValueError(
                "至少需要提供一个NMR模态（C或H），或者提供 input_ids："
                f"C-NMR (enabled={self.c_encoder is not None}, provided={c_peaks is not None}), "
                f"H-NMR (enabled={self.h_encoder is not None}, provided={h_peaks is not None})"
            )
            
        # ... Legacy Logic follows ...

        # ---------------------------------------------------------
        # Original Generation Logic (adapted)
        # ---------------------------------------------------------
        
            
        # If no input_ids provided, run specific encoding logic
        if input_ids is None:
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
                elif h_peaks is not None:
                    if h_peaks.dim() == 1:
                        h_peaks = h_peaks.unsqueeze(0)
                    batch_size = h_peaks.shape[0]
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
            if h_peaks is not None:
                h_peaks = h_peaks.to(device).float()
            if formula_vector is not None:
                formula_vector = formula_vector.to(device).float()
            if c_nmr_mask is not None:
                c_nmr_mask = c_nmr_mask.to(device).long()
            if h_nmr_mask is not None:
                h_nmr_mask = h_nmr_mask.to(device).long()
            
            # 编码NMR数据（根据配置）
            z_c = None
            if self.c_encoder is not None and c_peaks is not None:
                z_c = self.c_encoder(c_peaks, mask=c_nmr_mask)
            
            z_h = None
            if self.h_encoder is not None and h_peaks is not None:
                z_h = self.h_encoder(h_peaks, mask=h_nmr_mask)
            
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
            
            # 融合所有模态
            encoder_hidden = self.fusion(z_c, z_h, z_guidance)
            
            # 创建encoder输出
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
            
            # 创建attention mask
            batch_size_actual = encoder_hidden.shape[0]
            seq_len = encoder_hidden.shape[1]
            mask_list = []
            if c_nmr_mask is not None:
                mask_list.append(c_nmr_mask)
            if h_nmr_mask is not None:
                mask_list.append(h_nmr_mask)

            if len(mask_list) > 0:
                attention_mask = torch.cat(mask_list, dim=1)
                if z_guidance is not None:
                    formula_mask = torch.ones(batch_size_actual, 1, dtype=torch.long, device=device)
                    attention_mask = torch.cat([attention_mask, formula_mask], dim=1)
            else:
                attention_mask = torch.ones(batch_size_actual, seq_len, dtype=torch.long, device=device)

            if attention_mask.shape[1] != seq_len:
                if attention_mask.shape[1] > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                else:
                    padding = torch.zeros(
                        batch_size_actual, seq_len - attention_mask.shape[1],
                        dtype=torch.long, device=device
                    )
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
        
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
            tokens = tokens.cpu().numpy().tolist()
        elif hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        else:
            tokens = list(tokens)
        
        # Try using tokenizer's decode method if available (NMRMind has this)
        if hasattr(self.tokenizer, 'decode'):
            try:
                smiles = self.tokenizer.decode(tokens, skip_special_tokens=True)
                smiles = re.sub(r"\s+", "", smiles)
                return smiles
            except Exception:
                pass  # Fallback to manual method
        
        # Manual conversion for legacy tokenizer
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
            weight_decay=0.01
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






__all__ = ["NMR2SMILESModel", "PeakEncoder", "FusionLayer"]
