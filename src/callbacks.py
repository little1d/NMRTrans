import logging
from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

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
        
        # Print validation examples if available
        if hasattr(pl_module, "validation_outputs") and pl_module.validation_outputs:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Validation Examples (Epoch {epoch}):")
            logger.info(f"{'-' * 80}")
            
            for i, example in enumerate(pl_module.validation_outputs[:3]):  # Show first 3 examples
                logger.info(f"\nExample {i + 1}:")
                logger.info(f"  Original:  {example['original']}")
                logger.info(f"  Generated: {example['predicted_original']}")
                logger.info(f"  Token Acc: {example['val_token_acc']:.4f}")
                logger.info(f"  Seq Acc:   {example['val_seq_acc']:.4f}")
            
            logger.info(f"{'=' * 80}\n")
            
            # Clear validation outputs after printing
            pl_module.validation_outputs = []


__all__ = ["EpochResultPrinter", "ValidationResultPrinter"]
