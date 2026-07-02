import tempfile
import unittest
from types import SimpleNamespace

from src.callbacks import BestModelCheckpoint, get_default_callbacks


class CheckpointCallbackTests(unittest.TestCase):
    def test_best_checkpoint_saves_only_after_validation_without_version_counter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            callback = BestModelCheckpoint(dirpath=tmp_dir)

            self.assertFalse(callback._save_on_train_epoch_end)
            self.assertFalse(callback._enable_version_counter)
            self.assertEqual(callback.monitor, "val_seq_acc")
            self.assertEqual(callback.mode, "max")

    def test_default_callbacks_use_single_validation_checkpoint_callback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            callbacks = get_default_callbacks(SimpleNamespace(USE_SWANLAB=False), tmp_dir)
            checkpoint_callbacks = [
                callback for callback in callbacks if isinstance(callback, BestModelCheckpoint)
            ]

            self.assertEqual(len(checkpoint_callbacks), 1)
            checkpoint = checkpoint_callbacks[0]
            self.assertFalse(checkpoint._save_on_train_epoch_end)
            self.assertFalse(checkpoint._enable_version_counter)
            self.assertEqual(checkpoint.save_top_k, 3)


if __name__ == "__main__":
    unittest.main()
