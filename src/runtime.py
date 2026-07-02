"""Runtime helpers for configuring PyTorch Lightning training."""


def build_trainer_kwargs(config):
    """Return PyTorch Lightning Trainer keyword arguments from config."""
    return {
        "max_epochs": config.EPOCHS,
        "accelerator": config.ACCELERATOR,
        "devices": config.DEVICES,
        "strategy": config.STRATEGY,
        "precision": config.PRECISION,
        "gradient_clip_val": config.GRAD_CLIP,
        "accumulate_grad_batches": config.ACCUM_GRAD_BATCHES,
        "check_val_every_n_epoch": config.CHECK_VAL_EVERY_N_EPOCH,
        "limit_val_batches": config.LIMIT_VAL_BATCHES,
        "num_sanity_val_steps": config.NUM_SANITY_VAL_STEPS,
        "log_every_n_steps": config.LOG_EVERY_N_STEPS,
        "enable_progress_bar": config.ENABLE_PROGRESS_BAR,
        "enable_model_summary": config.ENABLE_MODEL_SUMMARY,
        "deterministic": config.DETERMINISTIC,
    }


def _requested_gpu_count(devices):
    if isinstance(devices, int):
        if devices == -1:
            return None
        return devices

    if isinstance(devices, (list, tuple)):
        return len(devices)

    if isinstance(devices, str):
        normalized = devices.strip().lower()
        if normalized in {"auto", "-1"}:
            return None
        if normalized.isdigit():
            return int(normalized)
        if "," in normalized:
            return len([item for item in normalized.split(",") if item.strip()])

    return None


def validate_runtime_config(config, cuda_available=None, cuda_device_count=None):
    """Fail early when GPU configuration cannot be satisfied."""
    accelerator = str(config.ACCELERATOR).lower()
    if accelerator not in {"gpu", "cuda"}:
        return

    if cuda_available is None or cuda_device_count is None:
        import torch

        if cuda_available is None:
            cuda_available = torch.cuda.is_available()
        if cuda_device_count is None:
            cuda_device_count = torch.cuda.device_count()

    if not cuda_available:
        raise RuntimeError(
            f"ACCELERATOR={config.ACCELERATOR!r} requires GPU training, but CUDA is not available."
        )

    requested = _requested_gpu_count(config.DEVICES)
    if requested is not None and requested > cuda_device_count:
        raise RuntimeError(
            f"Runtime config requires {requested} GPU device(s), but only "
            f"{cuda_device_count} visible device(s) are available. Check DEVICES "
            "or CUDA_VISIBLE_DEVICES."
        )
