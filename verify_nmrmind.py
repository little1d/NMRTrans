import sys
import os
import torch
import numpy as np

# Adjust path
sys.path.append(os.path.abspath("src"))

from config import TrainingConfig
from tokenizer_nmrmind import NMRMindTokenizer
from train import nmrmind_collate_fn
from model import NMR2SMILESModel

def test_pipeline():
    print("Testing NMRMind Pipeline...")
    
    # 1. Config
    config = TrainingConfig()
    config.TOKENIZER_TYPE = "nmrmind"
    vocab_path = os.path.abspath("vocab_nmrmind.json")
    config.VOCAB_NMRMIND_PATH = vocab_path
    
    print(f"Loading tokenizer from {vocab_path}")
    if not os.path.exists(vocab_path):
        print("Vocab file not found! creating dummy for test")
        # create dummy vocab
        import json
        dummy_vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "C_100.0": 4, "H_7.25": 5, "C": 6, "c": 7, "1": 8, "0": 9, "<1H_NMR>": 10, "</1H_NMR>": 11, "<13C_NMR>": 12, "</13C_NMR>": 13, "<molecular_formula>": 14, "</molecular_formula>": 15}
        with open("vocab_nmrmind_dummy.json", "w") as f:
            json.dump(dummy_vocab, f)
        vocab_path = "vocab_nmrmind_dummy.json"
        
    tokenizer = NMRMindTokenizer(vocab_path)
    print(f"Tokenizer loaded. Size: {len(tokenizer)}")
    
    # 2. Dummy Batch
    batch = [{
        "h_nmr_peaks": [7.25, 7.30],
        "c_nmr_peaks": [100.0, 120.5],
        "original_smiles": "Cc1ccccc1",
        "molecular_formula": "C7H8"
    }, {
        "h_nmr_peaks": [1.0],
        "c_nmr_peaks": [20.0],
        "original_smiles": "C",
        "molecular_formula": "CH4"
    }]
    
    print("Collating batch...")
    collated = nmrmind_collate_fn(batch, tokenizer, config)
    
    print("Keys in collated:", collated.keys())
    print("input_ids shape:", collated["input_ids"].shape)
    print("attention_mask shape:", collated["attention_mask"].shape)
    print("smiles (labels) shape:", collated["smiles"].shape)
    
    # 3. Model
    print("Initializing Model...")
    model = NMR2SMILESModel(config, tokenizer)
    model.eval()
    
    # 4. Forward
    print("Running Forward Pass...")
    with torch.no_grad():
        output = model.training_step(collated, 0)
        
    print("Forward Pass Loss:", output)
    
    # 5. Generate
    print("Running Generation...")
    gen_ids = model.generate(
        input_ids=collated["input_ids"],
        attention_mask=collated["attention_mask"]
    )
    print("Generated IDs shape:", gen_ids.shape)
    print("Decoded:", tokenizer.decode(gen_ids[0].tolist()))
    
    print("SUCCESS")

if __name__ == "__main__":
    test_pipeline()
