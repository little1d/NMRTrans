import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.config import PROJECT_ROOT, TrainingConfig, load_training_config


class TrainingConfigYamlTests(unittest.TestCase):
    def test_defaults_without_local_config_are_project_relative(self):
        config = TrainingConfig(local_config=None)

        self.assertEqual(config.MERGED_DATA_DIR, str(PROJECT_ROOT / "cache"))
        self.assertEqual(config.TRAIN_FILE, str(PROJECT_ROOT / "cache" / "train.pkl.lz4"))
        self.assertEqual(config.VAL_FILE, str(PROJECT_ROOT / "cache" / "val.pkl.lz4"))
        self.assertEqual(config.TEST_FILE, str(PROJECT_ROOT / "cache" / "test.pkl.lz4"))
        self.assertEqual(config.VOCAB_PATH, str(PROJECT_ROOT / "vocab.json"))
        self.assertEqual(config.SAVE_DIR, str(PROJECT_ROOT / "checkpoints"))
        self.assertEqual(config.T5_MODEL_NAME, str(PROJECT_ROOT / "models" / "t5-small"))

    def test_config_module_does_not_embed_legacy_absolute_paths(self):
        config_source = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")

        self.assertNotIn("/mnt/shared-storage-user", config_source)
        self.assertNotIn("yangzhuo", config_source)

    def test_yaml_overrides_values_and_refreshes_derived_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            config_path = base_dir / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "MERGED_DATA_DIR: cache",
                        "VOCAB_PATH: vocab.json",
                        "SAVE_DIR: checkpoints",
                        "T5_MODEL_NAME: models/t5-small",
                        "DEVICES: 3",
                        "BATCH_SIZE: 16",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_training_config(config_path, base_dir=base_dir)

            self.assertEqual(config.MERGED_DATA_DIR, str(base_dir / "cache"))
            self.assertEqual(config.TRAIN_FILE, str(base_dir / "cache" / "train.pkl.lz4"))
            self.assertEqual(config.VAL_FILE, str(base_dir / "cache" / "val.pkl.lz4"))
            self.assertEqual(config.TEST_FILE, str(base_dir / "cache" / "test.pkl.lz4"))
            self.assertEqual(config.VOCAB_PATH, str(base_dir / "vocab.json"))
            self.assertEqual(config.SAVE_DIR, str(base_dir / "checkpoints"))
            self.assertEqual(config.T5_MODEL_NAME, str(base_dir / "models" / "t5-small"))
            self.assertEqual(config.PEAK_ENCODER_D_MODEL, 512)
            self.assertEqual(config.DEVICES, 3)
            self.assertEqual(config.BATCH_SIZE, 16)

    def test_yaml_config_does_not_inherit_config_local_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            config_path = base_dir / "config.yaml"
            config_path.write_text("DEVICES: 3\n", encoding="utf-8")
            local_config = SimpleNamespace(BATCH_SIZE=7, USE_SWANLAB=True)

            config = load_training_config(
                config_path,
                base_dir=base_dir,
                local_config=local_config,
            )

            self.assertEqual(config.DEVICES, 3)
            self.assertEqual(config.BATCH_SIZE, 1024)
            self.assertFalse(config.USE_SWANLAB)

    def test_config_local_is_used_when_yaml_is_absent(self):
        local_config = SimpleNamespace(BATCH_SIZE=7, USE_SWANLAB=True)

        config = load_training_config(config_path=None, local_config=local_config)

        self.assertEqual(config.BATCH_SIZE, 7)
        self.assertTrue(config.USE_SWANLAB)

    def test_yaml_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            config_path = base_dir / "config.yaml"
            config_path.write_text("NOT_A_REAL_CONFIG: true\n", encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "Unknown config key"):
                load_training_config(config_path, base_dir=base_dir)


if __name__ == "__main__":
    unittest.main()
