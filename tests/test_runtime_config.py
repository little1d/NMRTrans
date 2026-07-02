import unittest
from types import SimpleNamespace

from src.config import TrainingConfig


class RuntimeConfigTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = {
            "EPOCHS": 7,
            "ACCELERATOR": "gpu",
            "DEVICES": 8,
            "STRATEGY": "ddp_find_unused_parameters_true",
            "PRECISION": "32",
            "GRAD_CLIP": 1.0,
            "ACCUM_GRAD_BATCHES": 2,
            "CHECK_VAL_EVERY_N_EPOCH": 3,
            "LIMIT_VAL_BATCHES": 0.5,
            "NUM_SANITY_VAL_STEPS": 0,
            "LOG_EVERY_N_STEPS": 25,
            "ENABLE_PROGRESS_BAR": False,
            "ENABLE_MODEL_SUMMARY": False,
            "DETERMINISTIC": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_training_config_exposes_runtime_defaults(self):
        self.assertTrue(hasattr(TrainingConfig, "ACCELERATOR"))
        self.assertTrue(hasattr(TrainingConfig, "STRATEGY"))
        self.assertTrue(hasattr(TrainingConfig, "NUM_SANITY_VAL_STEPS"))
        self.assertTrue(hasattr(TrainingConfig, "LOG_EVERY_N_STEPS"))
        self.assertTrue(hasattr(TrainingConfig, "ENABLE_PROGRESS_BAR"))
        self.assertTrue(hasattr(TrainingConfig, "ENABLE_MODEL_SUMMARY"))
        self.assertTrue(hasattr(TrainingConfig, "DETERMINISTIC"))

    def test_build_trainer_kwargs_uses_runtime_config(self):
        from src.runtime import build_trainer_kwargs

        kwargs = build_trainer_kwargs(self.make_config())

        self.assertEqual(
            kwargs,
            {
                "max_epochs": 7,
                "accelerator": "gpu",
                "devices": 8,
                "strategy": "ddp_find_unused_parameters_true",
                "precision": "32",
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 2,
                "check_val_every_n_epoch": 3,
                "limit_val_batches": 0.5,
                "num_sanity_val_steps": 0,
                "log_every_n_steps": 25,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "deterministic": True,
            },
        )

    def test_gpu_validation_rejects_insufficient_devices(self):
        from src.runtime import validate_runtime_config

        with self.assertRaisesRegex(RuntimeError, "requires 8 GPU"):
            validate_runtime_config(
                self.make_config(ACCELERATOR="gpu", DEVICES=8),
                cuda_available=True,
                cuda_device_count=4,
            )

    def test_gpu_validation_skips_cpu_accelerator(self):
        from src.runtime import validate_runtime_config

        validate_runtime_config(
            self.make_config(ACCELERATOR="cpu", DEVICES=1),
            cuda_available=False,
            cuda_device_count=0,
        )


if __name__ == "__main__":
    unittest.main()
