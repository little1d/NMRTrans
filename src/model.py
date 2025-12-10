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
    输入: (B, L) 的 ppm list → 输出: (B, d_model) 的全局语义向量
    """
    def __init__(self, d_model=512, n_layers=2, n_heads=4, ff_dim=1024):
        super().__init__()
        
        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=ff_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, peaks):
        """
        peaks: list of lists → 已转换成 padded tensor (B, L) or (B, L, 1)
        returns: (B, d_model)
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

        # input projection
        x = self.input_proj(x)    # (B, L, d_model)

        # transformer encoding
        x = self.transformer(x)   # (B, L, d_model)

        # mean pooling (avoid padding=0)
        mask = (peaks.squeeze(-1) != 0).unsqueeze(-1)  # padding=0
        x_sum = (x * mask).sum(dim=1)
        x_cnt = mask.sum(dim=1).clamp(min=1)

        x_mean = x_sum / x_cnt
        return x_mean   # (B, d_model)


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
        nmr_list = []

        if z_c is not None:
            nmr_list.append(z_c)
        if z_h is not None:
            nmr_list.append(z_h)

        if len(nmr_list) == 0:
            raise ValueError("No NMR modality provided! Need at least C or H.")

        # 融合主模态
        z_nmr = torch.stack(nmr_list, dim=0).mean(dim=0)   # (B, d_model)

        # 指导模态（formula/MS等）作为 bias
        if z_guidance is not None:
            z_all = z_nmr + z_guidance
        else:
            z_all = z_nmr

        # 注意 T5 encoder_outputs 需要 (B, L, d)
        # 我们这里 L=1
        return z_all.unsqueeze(1)   # (B, 1, d_model)


# ===========================================================
# 3. 主模型：NMR → SMILES（T5 decoder-only）
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
        
        # 主模态 encoder
        self.c_encoder = PeakEncoder(
            d_model=d_model,
            n_layers=config.PEAK_ENCODER_N_LAYERS,
            n_heads=config.PEAK_ENCODER_N_HEADS,
            ff_dim=config.PEAK_ENCODER_FF_DIM
        )
        self.h_encoder = PeakEncoder(
            d_model=d_model,
            n_layers=config.PEAK_ENCODER_N_LAYERS,
            n_heads=config.PEAK_ENCODER_N_HEADS,
            ff_dim=config.PEAK_ENCODER_FF_DIM
        )

        # Fusion
        self.fusion = FusionLayer(d_model=d_model)

        # T5 用于 decoder-only（但依然 load 整个模型）
        logger.info(f"Loading T5 model from: {config.T5_MODEL_NAME}")
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            config.T5_MODEL_NAME,
            local_files_only=True  # Use local model files only, don't access network
        )
        
        # Optionally freeze T5 decoder
        if config.FREEZE_T5_DECODER:
            logger.info("Freezing T5 decoder parameters")
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
        
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

        # future guidance (not implemented yet)
        z_guidance = None

        # 融合
        encoder_hidden = self.fusion(z_c, z_h, z_guidance)  # (B,1,d)

        # 构造 encoder_outputs（跳过 T5 encoder）
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        # 送入 T5 decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=smiles_ids,
            attention_mask=None,              # we don't need it for encoder
            decoder_attention_mask=attention_mask,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step for autoregressive generation."""
        smiles_ids = batch["smiles"].long()
        c_peaks = batch.get("c_nmr_peaks")
        h_peaks = batch.get("h_nmr_peaks")
        
        # T5 expects labels with -100 for positions to ignore
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        # Forward pass
        outputs = self(
            c_peaks=c_peaks,
            h_peaks=h_peaks,
            smiles_ids=labels
        )
        
        loss = outputs.loss
        
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
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_token_acc", token_acc, prog_bar=True, sync_dist=True)
        self.log("train_seq_acc", seq_acc, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with autoregressive generation."""
        smiles_ids = batch["smiles"].long()
        c_peaks = batch.get("c_nmr_peaks")
        h_peaks = batch.get("h_nmr_peaks")
        original_smiles_list = batch["original_smiles"]
        
        # Teacher forcing evaluation
        labels = smiles_ids.clone()
        labels[labels == self.config.PAD_TOKEN_ID] = -100
        
        with torch.no_grad():
            outputs = self(
                c_peaks=c_peaks,
                h_peaks=h_peaks,
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
        
        # Log metrics
        self.log("val_token_acc", token_acc, prog_bar=True, sync_dist=True)
        self.log("val_seq_acc", seq_acc, prog_bar=True, sync_dist=True)
        
        # Generate and log examples
        if batch_idx % 10 == 0 and self.global_rank == 0:
            generated = self.generate(c_peaks[:1], h_peaks[:1] if h_peaks is not None else None)
            generated_smiles = self.tokens_to_smiles(generated[0])
            original_smiles = original_smiles_list[0]
            
            logger.info(f"Original:  {original_smiles}")
            logger.info(f"Generated: {generated_smiles}")
            
            self.validation_outputs.append({
                "original": original_smiles,
                "predicted_original": generated_smiles,
                "val_token_acc": token_acc.item(),
                "val_seq_acc": seq_acc.item(),
            })
        
        return {"val_token_acc": token_acc, "val_seq_acc": seq_acc}
    
    def generate(self, c_peaks=None, h_peaks=None, max_length=None):
        """Generate SMILES using T5 generation."""
        if max_length is None:
            max_length = self.config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
        
        # Encode spectra
        with torch.no_grad():
            z_c = self.c_encoder(c_peaks) if c_peaks is not None else None
            z_h = self.h_encoder(h_peaks) if h_peaks is not None else None
            encoder_hidden = self.fusion(z_c, z_h, None)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
            
            # Use T5 generate method
            generated_ids = self.t5.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.config.PAD_TOKEN_ID,
                eos_token_id=self.config.EOS_TOKEN_ID,
                bos_token_id=self.config.BOS_TOKEN_ID,
            )
        
        return generated_ids
    
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
