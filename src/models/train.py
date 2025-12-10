# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from model import NMR2SMILESModel  
# TODO: 替换为你实际 dataset module 名称
from data import NMRPeakDataset  

def collate_fn(batch):
    """Batch collate，把 peaks list pad成 tensor。"""
    # 假设所有 peak 列表是 list[float]，不同样本长度可能不同
    from torch.nn.utils.rnn import pad_sequence
    import torch

    c_peaks = [torch.tensor(b["c_nmr_peaks"], dtype=torch.float) for b in batch if b.get("c_nmr_peaks") is not None]
    h_peaks = [torch.tensor(b["h_nmr_peaks"], dtype=torch.float) for b in batch if b.get("h_nmr_peaks") is not None]
    # smiles_ids 已经是你自定义 tokenizer 输出
    smiles_ids = [torch.tensor(b["smiles"], dtype=torch.long) for b in batch]

    # pad peaks
    c_padded = pad_sequence(c_peaks, batch_first=True, padding_value=0.0) if c_peaks else None
    h_padded = pad_sequence(h_peaks, batch_first=True, padding_value=0.0) if h_peaks else None

    # pad smiles_ids to same length
    smiles_padded = pad_sequence(smiles_ids, batch_first=True, padding_value=-100)  # -100 for ignore_index

    return {
        "c_peaks": c_padded,
        "h_peaks": h_padded,
        "smiles_ids": smiles_padded,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="path to train lz4-pkl dataset")
    parser.add_argument("--valid_data", type=str, required=True, help="path to valid lz4-pkl dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--t5_model", type=str, default="t5-small")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    args = parser.parse_args()

    train_ds = NMRPeakDataset(args.train_data)
    valid_ds = NMRPeakDataset(args.valid_data)

    model = NMR2SMILESModel(t5_name=args.t5_model).cuda()

    def data_collator(batch):
        return collate_fn(batch)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )

    trainer.train()
    model_path = os.path.join(args.output_dir, "final_model")
    model.t5.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()