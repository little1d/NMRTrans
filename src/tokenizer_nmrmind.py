import json
import re
import numpy as np
from typing import List, Dict, Union, Optional

class NMRMindTokenizer:
    """
    Tokenizer using the vocab_nmrmind.json and supporting NMR spectra quantization/tokenization.
    """
    def __init__(self, vocab_file: str):
        with open(vocab_file, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # SMILES Regex Pattern (same as original)
        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        
        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        
        self.pad_token_id = self.token_to_id.get(self.pad_token, 1)
        self.bos_token_id = self.token_to_id.get(self.bos_token, 0)
        self.eos_token_id = self.token_to_id.get(self.eos_token, 2)
        self.unk_token_id = self.token_to_id.get(self.unk_token, 3)
        self.mask_token_id = self.token_to_id.get(self.mask_token, None) # Standard mask token
        
        # NMR Tags based on vocab analysis
        self.tag_13c_start = "<13C_NMR>" # ID 4026
        self.tag_13c_end = "</13C_NMR>" # ID 4027
        self.tag_1h_start = "<1H_NMR>"  # ID 4028
        self.tag_1h_end = "</1H_NMR>"   # ID 4029
        
    def __len__(self):
        return len(self.token_to_id)

    def tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string using regex."""
        return re.findall(self.pattern, smiles)

    def encode_smiles(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize_smiles(smiles)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return self.convert_tokens_to_ids(tokens)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.token_to_id.get(tokens, self.unk_token_id)
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.id_to_token.get(ids, self.unk_token)
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self.pad_token, self.bos_token, self.eos_token, self.tag_13c_start, self.tag_13c_end, self.tag_1h_start, self.tag_1h_end]]
        return "".join(tokens)

    def quantize_c_nmr(self, peaks: List[float]) -> List[str]:
        """
        Quantize C-NMR peaks to tokens logic:
        - Range: 0.0 - 230.0
        - Resolution: 0.1
        - Format: C_{:.1f}
        """
        tokens = []
        if not peaks or len(peaks) == 0:
            return tokens
            
        for p in peaks:
            if np.isnan(p):
                tokens.append("C_NAN")
                continue
            
            # Round to nearest 0.1
            val = round(float(p), 1)
            
            # Clip
            if val < 0.0: val = 0.0
            if val > 230.0: val = 230.0
            
            token = f"C_{val:.1f}"
            if token in self.token_to_id:
                tokens.append(token)
            else:
                # Fallback? Should not happen if vocab is complete for this range
                # Try finding nearest?
                pass 
        return tokens

    def quantize_h_nmr(self, peaks: List[float]) -> List[str]:
        """
        Quantize H-NMR peaks to tokens logic:
        - Range: 0.00 - 15.00
        - Resolution: 0.01
        - Format: H_{:.2f}
        """
        tokens = []
        if not peaks or len(peaks) == 0:
            return tokens
            
        for p in peaks:
            if np.isnan(p):
                tokens.append("H_NAN")
                continue
            
            # Round to nearest 0.01
            val = round(float(p), 2)
            
            # Clip
            if val < 0.00: val = 0.00
            if val > 15.00: val = 15.00
            
            token = f"H_{val:.2f}"
            if token in self.token_to_id:
                tokens.append(token)
            else:
                pass
        return tokens

    def encode_spectra(self, c_peaks: Optional[List[float]], h_peaks: Optional[List[float]]) -> List[int]:
        """
        Encodes spectra into a single sequence of token IDs:
        Format: <1H_NMR> H_x.xx ... </1H_NMR> <13C_NMR> C_x.x ... </13C_NMR> 
        (Order can be flexible, matching nmr_main usage)
        """
        input_tokens = []
        
        # H-NMR Section
        if h_peaks is not None and len(h_peaks) > 0:
            input_tokens.append(self.tag_1h_start)
            h_tokens = self.quantize_h_nmr(h_peaks)
            # Sort peaks? Usually good practice, though transformers might handle unsorted
            # String sorting might be weird (H_10 < H_2), so better sort floats then quantize
            # Assuming input is already sorted or we sort here. 
            # Simple lexicographical sort of tokens might not correspond to value sort perfectly
            # but let's trust the quantization function inputs are pre-sorted or don't matter enough.
            # Ideally:
            input_tokens.extend(h_tokens)
            input_tokens.append(self.tag_1h_end)
            
        # C-NMR Section
        if c_peaks is not None and len(c_peaks) > 0:
            input_tokens.append(self.tag_13c_start)
            c_tokens = self.quantize_c_nmr(c_peaks)
            input_tokens.extend(c_tokens)
            input_tokens.append(self.tag_13c_end)
            
        return self.convert_tokens_to_ids(input_tokens)
