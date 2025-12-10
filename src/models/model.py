import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, BaseModelOutput


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
        peaks: list of lists → 已转换成 padded tensor (B, L)
        returns: (B, d_model)
        """
        if peaks is None:
            return None

        # (B, L) -> (B, L, 1)
        x = peaks.unsqueeze(-1).float()

        # input projection
        x = self.input_proj(x)    # (B, L, d_model)

        # transformer encoding
        x = self.transformer(x)   # (B, L, d_model)

        # mean pooling
        mask = (peaks != 0).unsqueeze(-1)  # padding=0
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

class FormulaEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 假设 formula 已被 parsed 为元素计数 dict / vector
        self.proj = nn.Linear(num_element_types, d_model)
    def forward(self, formula_vec):
        return self.proj(formula_vec)  # (B, d_model)

# ===========================================================
# 3. 主模型：NMR → SMILES（T5 decoder-only）
# ===========================================================

class NMR2SMILESModel(nn.Module):
    """
    主架构：
    - 不使用 T5 encoder
    - 自己实现 H/C encoder
    - guidance 接口保留但不实现
    - 最终输出喂给 T5 decoder
    """
    def __init__(self, t5_name="t5-small", d_model=512):
        super().__init__()

        # 主模态 encoder
        self.c_encoder = PeakEncoder(d_model=d_model)
        self.h_encoder = PeakEncoder(d_model=d_model)
        self.formula_encoder = FormulaEncoder(d_model)

        # Fusion
        self.fusion = FusionLayer(d_model=d_model)

        # T5 用于 decoder-only（但依然 load 整个模型）
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)

        # future modal encoders（暂时为 None）
        self.use_formula = False
        self.use_ms = False

    # =======================================================
    # 占位接口（未来实现）
    # =======================================================
    def encode_formula(self, formula_str):
        return None

    def encode_ms(self, ms_data):
        return None

    # =======================================================
    # forward
    # =======================================================

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

        # future guidance
        z_guidance = None
        if formula is not None:
            z_guidance = self.formula_encoder(formula)

        # 融合
        encoder_hidden = self.fusion(z_c, z_h, z_guidance)  # (B,1,d)

        # 构造 encoder_outputs（跳过 T5 encoder）
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        # 送入 T5 decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=smiles_ids,
            attention_mask=None,              # we don't need it for encoder
            decoder_attention_mask=None,
        )
        return outputs