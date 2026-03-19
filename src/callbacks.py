import logging
import os
from typing import Any, Dict
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

logger = logging.getLogger(__name__)


class EpochResultPrinter(Callback):
    """Callback to print training metrics at the end of each epoch."""
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print training metrics at the end of each training epoch."""
        if trainer.global_rank != 0:
            return  # Only print on rank 0
        
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Extract training metrics
        train_metrics = {}
        for key in ["train_loss", "train_token_acc", "train_seq_acc", "lr"]:
            if key in metrics:
                train_metrics[key] = metrics[key].item() if hasattr(metrics[key], "item") else metrics[key]
        
        if train_metrics:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch} Training Results:")
            logger.info(f"{'-' * 80}")
            for key, value in train_metrics.items():
                if "acc" in key.lower():
                    logger.info(f"  {key}: {value:.4f}")
                elif key == "lr":
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value:.6f}")
            logger.info(f"{'=' * 80}\n")


class ValidationResultPrinter(Callback):
    """Callback to print validation metrics and examples after validation."""
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print validation metrics at the end of validation."""
        if trainer.global_rank != 0:
            return  # Only print on rank 0
        
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Extract validation metrics
        val_metrics = {}
        for key in metrics:
            if key.startswith("val_"):
                val_metrics[key] = metrics[key].item() if hasattr(metrics[key], "item") else metrics[key]
        
        if val_metrics:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch} Validation Results:")
            logger.info(f"{'-' * 80}")
            for key, value in sorted(val_metrics.items()):
                if "acc" in key.lower():
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value:.6f}")
            logger.info(f"{'=' * 80}\n")
        
        # ✅ 修复：安全打印验证示例（兼容新旧格式）
        if hasattr(pl_module, "validation_outputs") and pl_module.validation_outputs:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Validation Examples (Epoch {epoch}):")
            logger.info(f"{'-' * 80}")
            
            examples_printed = 0
            for example in pl_module.validation_outputs:
                if examples_printed >= 3:  # Show first 3 examples
                    break
                
                # ✅ 安全访问：兼容新旧格式
                original = example.get("original", example.get("true_smiles", "N/A"))
                predicted = example.get("predicted_original", example.get("pred_smiles", "N/A"))
                token_acc = example.get("val_token_acc", example.get("token_acc", 0.0))
                seq_acc = example.get("val_seq_acc", example.get("seq_acc", 0.0))
                
                # 只打印有有效内容的示例
                if original != "N/A" or predicted != "N/A":
                    logger.info(f"\nExample {examples_printed + 1}:")
                    logger.info(f"  Original:  {original}")
                    logger.info(f"  Generated: {predicted}")
                    logger.info(f"  Token Acc: {token_acc:.4f}")
                    logger.info(f"  Seq Acc:   {seq_acc:.4f}")
                    examples_printed += 1
            
            if examples_printed == 0:
                logger.info("  (No examples available for display)")
            
            logger.info(f"{'=' * 80}\n")
            
            # ✅ 清空 outputs，避免内存累积
            pl_module.validation_outputs = []


class BestModelCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback with best model tracking and SwanLab integration."""
    
    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_seq_acc",
        mode: str = "max",
        save_top_k: int = 3,
        filename: str = "full-ar-{epoch:02d}-valacc={val_seq_acc:.4f}",
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
            save_weights_only=False,
            every_n_epochs=5,
            save_on_train_epoch_end=True,
            **kwargs
        )
        self.best_model_path = None
        self.best_model_score = None
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Track best model and log to SwanLab."""
        super().on_validation_end(trainer, pl_module)
        
        if trainer.global_rank != 0:
            return
        
        # ✅ 修复：正确比较 best_model_score
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            current_score_val = current_score.item() if hasattr(current_score, "item") else current_score
            
            if self.best_model_score is None or (
                (self.mode == "max" and current_score_val > self.best_model_score) or
                (self.mode == "min" and current_score_val < self.best_model_score)
            ):
                self.best_model_score = current_score_val
                if self.best_model_path:
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"New best model saved!")
                    logger.info(f"  Path: {self.best_model_path}")
                    logger.info(f"  {self.monitor}: {self.best_model_score:.4f}")
                    logger.info(f"{'=' * 80}\n")
                
                # Log to SwanLab if available
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    try:
                        if hasattr(trainer.logger, 'experiment'):
                            trainer.logger.experiment.log({
                                "best_model_path": self.best_model_path or "N/A",
                                f"best_{self.monitor}": self.best_model_score,
                            })
                    except Exception as e:
                        logger.warning(f"Failed to log best model to SwanLab: {e}")


class SwanLabImageLogger(Callback):
    """Callback to log images and visualizations to SwanLab."""
    
    def __init__(self, log_every_n_epochs: int = 10):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log validation examples to SwanLab."""
        if trainer.global_rank != 0:
            return
        
        epoch = trainer.current_epoch
        
        # Only log every N epochs
        if epoch % self.log_every_n_epochs != 0:
            return
        
        # Check if SwanLab is available
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            return
        
        if not hasattr(trainer.logger, 'experiment'):
            return
        
        # ✅ 修复：安全记录验证示例
        if hasattr(pl_module, "validation_outputs") and pl_module.validation_outputs:
            try:
                import swanlab
                
                examples_data = []
                for example in pl_module.validation_outputs[:5]:
                    # ✅ 安全访问键
                    original = example.get("original", example.get("true_smiles", "N/A"))
                    predicted = example.get("predicted_original", example.get("pred_smiles", "N/A"))
                    token_acc = example.get("val_token_acc", example.get("token_acc", 0.0))
                    seq_acc = example.get("val_seq_acc", example.get("seq_acc", 0.0))
                    
                    if original != "N/A" or predicted != "N/A":
                        examples_data.append({
                            "idx": len(examples_data) + 1,
                            "original": original,
                            "predicted": predicted,
                            "token_acc": f"{token_acc:.4f}",
                            "seq_acc": f"{seq_acc:.4f}",
                        })
                
                if examples_data:
                    # Log as table
                    trainer.logger.experiment.log({
                        f"validation_examples_epoch_{epoch}": swanlab.Table(
                            columns=["idx", "original", "predicted", "token_acc", "seq_acc"],
                            data=[[d["idx"], d["original"], d["predicted"], d["token_acc"], d["seq_acc"]] 
                                  for d in examples_data]
                        )
                    })
                    logger.info(f"Logged {len(examples_data)} validation examples to SwanLab")
                
            except ImportError:
                pass  # swanlab not installed
            except Exception as e:
                logger.warning(f"Failed to log examples to SwanLab: {e}")


def get_default_callbacks(config, save_dir: str):
    """Get default callbacks for training."""
    callbacks = [
        EpochResultPrinter(),
        ValidationResultPrinter(),
        GradientMonitor(log_every_n_steps=50),
        BestModelCheckpoint(
            dirpath=save_dir,
            monitor="val_seq_acc",
            mode="max",
            save_top_k=3,
            filename="ar-{epoch:02d}-valacc={val_seq_acc:.4f}",
        ),
    ]
    
    if getattr(config, "USE_SWANLAB", False):
        callbacks.append(SwanLabImageLogger(log_every_n_epochs=10))
    
    return callbacks


class GradientMonitor(Callback):
    """Monitor gradients and detect NaN/Inf issues."""
    
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check gradients after backward pass."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if trainer.global_rank != 0:
            return
        
        has_nan = False
        has_inf = False
        max_grad = 0.0
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan = True
                    logger.warning(f"NaN gradient detected in {name}")
                if torch.isinf(param.grad).any():
                    has_inf = True
                    logger.warning(f"Inf gradient detected in {name}")
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
        
        if has_nan or has_inf:
            logger.error(f"Gradient issues at step {trainer.global_step}: NaN={has_nan}, Inf={has_inf}")
        
        if trainer.logger:
            trainer.logger.log_metrics({
                "grad_max_norm": max_grad,
                "grad_has_nan": float(has_nan),
                "grad_has_inf": float(has_inf),
            }, step=trainer.global_step)


__all__ = [
    "EpochResultPrinter",
    "ValidationResultPrinter", 
    "BestModelCheckpoint",
    "SwanLabImageLogger",
    "GradientMonitor",
    "get_default_callbacks",
]