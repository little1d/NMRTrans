import logging
import re
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)


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
        
        # 4. Transformer 编码器（带 dropout）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=ff_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. 输出 LayerNorm
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, peaks):
        """
        peaks: list of lists → 已转换成 padded tensor (B, L) or (B, L, 1)
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
        x = self.transformer(x)   # (B, L, d_model)

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
    - guidance 接口保留但不实现
    - 最终输出喂给 T5 decoder
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        
        d_model = config.PEAK_ENCODER_D_MODEL
        
        # 主模态 encoder（添加 dropout 和 max_peaks 参数）
        self.c_encoder = PeakEncoder(
            d_model=d_model,
            n_layers=config.PEAK_ENCODER_N_LAYERS,
            n_heads=config.PEAK_ENCODER_N_HEADS,
            ff_dim=config.PEAK_ENCODER_FF_DIM,
            dropout=config.PEAK_ENCODER_DROPOUT,
            max_peaks=config.MAX_PEAKS
        )
        self.h_encoder = PeakEncoder(
            d_model=d_model,
            n_layers=config.PEAK_ENCODER_N_LAYERS,
            n_heads=config.PEAK_ENCODER_N_HEADS,
            ff_dim=config.PEAK_ENCODER_FF_DIM,
            dropout=config.PEAK_ENCODER_DROPOUT,
            max_peaks=config.MAX_PEAKS
        )

        
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
        logger.info(f"Loading T5 model from: {config.T5_MODEL_NAME}")
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            config.T5_MODEL_NAME,
            local_files_only=True  # Use local model files only, don't access network
        )
        
        # IMPORTANT: Set T5 to training mode
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
        formula_vector=None,
        **kwargs
    ):
        """
        输入:
          c_peaks: List[List[float]] → 已 padded tensor
          h_peaks: 同上
          smiles_ids: tokenized smiles (labels)
        """

        # C-NMR
        z_c = self.c_encoder(c_peaks) if c_peaks is not None else None

        # H-NMR
        z_h = self.h_encoder(h_peaks) if h_peaks is not None else None

        # 新增：处理化学式guidance
        z_guidance = None
        if self.formula_encoder is not None and formula_vector is not None:
            z_guidance = self.formula_encoder(formula_vector)  # (B, 1, d_model)
        
        # 融合
        encoder_hidden = self.fusion(z_c, z_h, z_guidance)  # (B, L_total, d_model)

        # 构造 encoder_outputs（跳过 T5 encoder）
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        # Create encoder attention mask for the peak sequence
        # encoder_hidden: (B, L, d_model) where L = num_c_peaks + num_h_peaks
        batch_size, seq_len, _ = encoder_hidden.shape
        
        # Create mask: all ones since all peaks are valid (no padding in peaks)
        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=encoder_hidden.device)

        # 送入 T5 decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,  # attention mask for encoder outputs
            labels=smiles_ids,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step for autoregressive generation."""
        smiles_ids = batch["smiles"].long()
        c_peaks = batch.get("c_nmr_peaks")
        h_peaks = batch.get("h_nmr_peaks")
        
        # Check for NaN in input data
        if c_peaks is not None and torch.isnan(c_peaks).any():
            logger.warning(f"NaN detected in c_peaks at batch {batch_idx}")
        if h_peaks is not None and torch.isnan(h_peaks).any():
            logger.warning(f"NaN detected in h_peaks at batch {batch_idx}")
        
        # T5 expects labels with -100 for positions to ignore
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None

        # Forward pass
        outputs = self(
            c_peaks=c_peaks,
            h_peaks=h_peaks,
            formula_vector=formula_vector,
            smiles_ids=labels
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
    
    def validation_step(self, batch, batch_idx):
        """Validation step with autoregressive generation."""
        if batch_idx == 0 and self.global_rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting validation at epoch {self.current_epoch}")
            logger.info(f"{'='*80}")
        
        smiles_ids = batch["smiles"].long()
        c_peaks = batch.get("c_nmr_peaks")
        h_peaks = batch.get("h_nmr_peaks")
        original_smiles_list = batch["original_smiles"]
        formula_vector = batch.get("formula_vector") if self.formula_encoder is not None else None

        # Teacher forcing evaluation
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        with torch.no_grad():
            outputs = self(
                c_peaks=c_peaks,
                h_peaks=h_peaks,
                formula_vector=formula_vector,
                smiles_ids=labels
            )
            
            logits = outputs.logits
            pred_tokens = logits.argmax(dim=-1)
            
            # Token accuracy
            valid_mask = (smiles_ids != self.config.PAD_TOKEN_ID)
            correct = (pred_tokens == smiles_ids) & valid_mask
            token_acc = correct.sum().float() / valid_mask.sum().float()
            
            # Sequence accuracy
            seq_correct = ((pred_tokens == smiles_ids) | ~valid_mask).all(dim=1)
            seq_acc = seq_correct.float().mean()
        
        # Log metrics (添加 logger=True 以同步到 SwanLab)
        self.log("val_token_acc", token_acc, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_seq_acc", seq_acc, prog_bar=True, sync_dist=True, logger=True)
        
        # Generate and log examples
        if batch_idx % 10 == 0 and self.global_rank == 0:
            try:
                generated = self.generate(c_peaks[:1], h_peaks[:1] if h_peaks is not None else None, formula_vector[:1] if formula_vector is not None else None)
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
    
    def generate(self, c_peaks=None, h_peaks=None, formula_vector=None, max_length=None, num_beams=1, 
                do_sample=False, temperature=1.0, top_k=50, top_p=1.0, batch_size=None):
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
        # 设置默认最大长度
        if max_length is None:
            max_length = self.config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
        
        # 检查是否有有效的输入
        has_nmr = (c_peaks is not None) or (h_peaks is not None)
        has_formula = self.formula_encoder is not None and formula_vector is not None
        
        if not (has_nmr or has_formula):
            raise ValueError("At least one of C-NMR peaks, H-NMR peaks, or formula vector must be provided")
        
        with torch.no_grad():
            # 确定设备
            device = next(self.parameters()).device
            
            # 处理单个样本情况（添加批次维度）
            if batch_size is None:
                if c_peaks is not None:
                    if c_peaks.dim() == 1:
                        c_peaks = c_peaks.unsqueeze(0)
                        batch_size = 1
                    else:
                        batch_size = c_peaks.shape[0]
                elif h_peaks is not None:
                    if h_peaks.dim() == 1:
                        h_peaks = h_peaks.unsqueeze(0)
                        batch_size = 1
                    else:
                        batch_size = h_peaks.shape[0]
                elif formula_vector is not None:
                    if formula_vector.dim() == 1:
                        formula_vector = formula_vector.unsqueeze(0)
                        batch_size = 1
                    else:
                        batch_size = formula_vector.shape[0]
            
            # 确保所有输入都在正确设备上
            if c_peaks is not None:
                c_peaks = c_peaks.to(device).float()
            if h_peaks is not None:
                h_peaks = h_peaks.to(device).float()
            if formula_vector is not None:
                formula_vector = formula_vector.to(device).float()
            
            # 编码NMR数据
            z_c = self.c_encoder(c_peaks) if c_peaks is not None else None
            z_h = self.h_encoder(h_peaks) if h_peaks is not None else None
            
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
                    logger.warning(f"Formula vector dimension mismatch: expected {expected_dim}, got {actual_dim}")
                    # 尝试裁剪或填充
                    if actual_dim < expected_dim:
                        padding = torch.zeros(formula_vector.shape[0], expected_dim - actual_dim, device=device)
                        formula_vector = torch.cat([formula_vector, padding], dim=1)
                    else:
                        formula_vector = formula_vector[:, :expected_dim]
                
                z_guidance = self.formula_encoder(formula_vector)
                logger.debug(f"Formula guidance encoded: shape {z_guidance.shape}")
            
            # 融合所有模态
            try:
                encoder_hidden = self.fusion(z_c, z_h, z_guidance)
                logger.debug(f"Fusion output shape: {encoder_hidden.shape}")
            except Exception as e:
                logger.error(f"Fusion failed: {str(e)}")
                logger.error(f"z_c shape: {z_c.shape if z_c is not None else None}")
                logger.error(f"z_h shape: {z_h.shape if z_h is not None else None}")
                logger.error(f"z_guidance shape: {z_guidance.shape if z_guidance is not None else None}")
                raise
            
            # 创建encoder输出
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
            
            # 创建attention mask
            batch_size_actual = encoder_hidden.shape[0]
            seq_len = encoder_hidden.shape[1]
            attention_mask = torch.ones(batch_size_actual, seq_len, dtype=torch.long, device=device)
            
            # 打印调试信息
            logger.debug(f"Generate input shapes:")
            logger.debug(f"  encoder_hidden: {encoder_hidden.shape}")
            logger.debug(f"  attention_mask: {attention_mask.shape}")
            logger.debug(f"  batch_size: {batch_size_actual}")
            logger.debug(f"  num_beams: {num_beams}")
            logger.debug(f"  do_sample: {do_sample}")
            
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
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
                
                logger.debug(f"Generated IDs shape: {generated_ids.shape}")
                return generated_ids
            
            except Exception as e:
                logger.error(f"T5 generation failed: {str(e)}")
                logger.error(f"encoder_outputs shape: {encoder_hidden.shape}")
                logger.error(f"attention_mask shape: {attention_mask.shape}")
                
                # 尝试更简单的生成方式作为回退
                try:
                    logger.warning("Attempting fallback generation with minimal parameters")
                    generated_ids = self.t5.generate(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        pad_token_id=self.config.PAD_TOKEN_ID,
                        eos_token_id=self.config.EOS_TOKEN_ID,
                        bos_token_id=self.config.BOS_TOKEN_ID,
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
        pad_count = 0
        
        for i, token in enumerate(tokens):
            if found_eos:
                if token == self.config.PAD_TOKEN_ID:
                    pad_count += 1
                    if pad_count == 1:
                        token_strings.append("<pad>")
                continue
            
            token_id = int(token)
            
            # 处理特殊token
            if token_id == self.config.EOS_TOKEN_ID:
                token_strings.append("<eos>")
                found_eos = True
                continue
            elif token_id == self.config.BOS_TOKEN_ID:
                token_strings.append("<bos>")
                continue
            elif token_id == self.config.PAD_TOKEN_ID:
                token_strings.append("<pad>")
                continue
            else:
                # 普通token
                try:
                    token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                    if isinstance(token_str, list):
                        token_str = token_str[0]
                    token_strings.append(token_str)
                except Exception:
                    # 如果tokenizer无法转换，显示未知token
                    token_strings.append(f"[UNK_{token_id}]")
        
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


__all__ = ["NMR2SMILESModel", "PeakEncoder", "FusionLayer"]
