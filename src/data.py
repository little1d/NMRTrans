"""Data loading utilities for Spectra2Smiles-AR."""

import logging
import pickle
import time
from torch.utils.data import Dataset
import lz4.frame as lz4

logger = logging.getLogger(__name__)


class MergedDataset(Dataset):
    """Memory resident dataset used once data shards have been merged."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = []
        self.total_samples = 0

        logger.info(f"正在加载数据集: {file_path}")
        start_time = time.time()

        try:
            if file_path.endswith(".lz4") or file_path.endswith(".pkl.lz4"):
                open_func = lz4.open
            else:
                open_func = open

            with open_func(file_path, "rb") as f:
                # 读取所有pickle对象
                while True:
                    try:
                        batch = pickle.load(f)
                        if isinstance(batch, list):
                            self.data.extend(batch)
                        else:
                            self.data.append(batch)
                    except EOFError:
                        break
                    except Exception as e:
                        logger.error(f"加载pickle对象时出错: {str(e)}")
                        break

            self.total_samples = len(self.data)
            load_time = time.time() - start_time
            logger.info(
                f"数据集加载完成，共 {self.total_samples} 个样本，耗时 {load_time:.2f} 秒"
            )
        except Exception as exc:
            logger.error(f"加载数据集 {file_path} 时出错: {exc}")
            raise

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        return self.data[idx]


__all__ = ["MergedDataset"]
