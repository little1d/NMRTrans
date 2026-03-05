"""Data loading utilities for Spectra2Smiles-AR."""

import logging
import pickle
import time
import json
import re
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import lz4.frame as lz4

logger = logging.getLogger(__name__)

class MergedDataset(Dataset):
    """
    New dataset class for handling the updated data format with tokenized_input field.
    
    Expected data format:
    - Each item should have: 'id', 'smiles', 'tokenized_input', 'atom_count', 'molecular_formula' (optional)
    - 'tokenized_input' is a JSON string that parses to a dict with:
        * "1HNMR": list of peaks, each peak is [chem_shift, peakwidth_str, split_str, integral_str, list_of_floats]
        * "13CNMR": list of float values (same format as original c_nmr_peaks)
    """

    def __init__(self, file_path: str, max_peaks=200):
        self.file_path = file_path
        self.data = []
        self.total_samples = 0
        self.max_peaks = max_peaks

        logger.info(f"正在加载新格式数据集: {file_path}")
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
                        continue

            self.total_samples = len(self.data)
            load_time = time.time() - start_time
            
            # 验证数据格式
            self._validate_data_format()
            
            logger.info(
                f"新格式数据集加载完成，共 {self.total_samples} 个样本，耗时 {load_time:.2f} 秒"
            )
        except Exception as exc:
            logger.error(f"加载新格式数据集 {file_path} 时出错: {exc}")
            raise

    def _validate_data_format(self):
        """验证数据格式是否符合新格式要求"""
        if not self.data:
            logger.warning("数据集为空，无法验证格式")
            return

        sample = self.data[0]
        required_fields = ['id', 'smiles', 'tokenized_input', 'atom_count']
        
        logger.info("验证新数据集格式:")
        logger.info(f"  样本示例: {sample}")
        
        for field in required_fields:
            if field in sample:
                logger.info(f"  ✅ 字段 '{field}' 存在")
            else:
                logger.warning(f"  ❌ 字段 '{field}' 不存在")
        
        if 'tokenized_input' in sample:
            try:
                tokenized_input = json.loads(sample['tokenized_input'])
                logger.info(f"  ✅ tokenized_input 可解析为JSON")
                logger.info(f"    包含的键: {list(tokenized_input.keys())}")
                
                if '1HNMR' in tokenized_input:
                    h_sample = tokenized_input['1HNMR'][:1] if tokenized_input['1HNMR'] else []
                    logger.info(f"      1HNMR 样本: {h_sample}")
                
                if '13CNMR' in tokenized_input:
                    c_sample = tokenized_input['13CNMR'][:5] if tokenized_input['13CNMR'] else []
                    logger.info(f"      13CNMR 样本: {c_sample}")
            except Exception as e:
                logger.warning(f"  ❌ tokenized_input 解析失败: {str(e)}")
                logger.info(f"    原始值: {sample['tokenized_input'][:100]}...")

    def _parse_tokenized_input(self, tokenized_input_str: str):
        """
        解析tokenized_input字段，提取H-NMR和C-NMR峰数据
        
        Returns:
            tuple: (h_nmr_peaks, c_nmr_peaks)
                h_nmr_peaks: list of list, each inner list has 5 elements [chem_shift, peakwidth_str, split_str, integral_str, j_coupling_list]
                c_nmr_peaks: list of float values
        """
        try:
            tokenized_input = json.loads(tokenized_input_str)
            
            h_nmr_peaks = []
            if "1HNMR" in tokenized_input and isinstance(tokenized_input["1HNMR"], list):
                # 保留所有5个特征，但peakwidth不再使用
                h_nmr_peaks = tokenized_input["1HNMR"]
            
            c_nmr_peaks = []
            if "13CNMR" in tokenized_input and isinstance(tokenized_input["13CNMR"], list):
                # 13CNMR直接是浮点数列表
                c_nmr_peaks = [float(peak) for peak in tokenized_input["13CNMR"] 
                             if isinstance(peak, (int, float))]
            
            return h_nmr_peaks, c_nmr_peaks
            
        except Exception as e:
            logger.warning(f"解析tokenized_input失败: {str(e)}")
            return [], []

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        try:
            item = self.data[idx]
            
            # 确保必要的字段存在
            if 'smiles' not in item:
                logger.warning(f"样本 {idx} 缺少 'smiles' 字段")
                return None
            
            if 'tokenized_input' not in item:
                logger.warning(f"样本 {idx} 缺少 'tokenized_input' 字段")
                return None
            
            # 解析tokenized_input
            h_nmr_peaks, c_nmr_peaks = self._parse_tokenized_input(item['tokenized_input'])
            
            # 创建与旧代码兼容的数据项
            processed_item = {
                'original_smiles': item['smiles'],  # 保持与旧代码兼容
                'smiles': item['smiles'],
                'id': item.get('id', f'sample_{idx}'),
                'atom_count': item.get('atom_count', 0),
                'tokenized_input': item['tokenized_input'],  # 保留原始字符串
                'h_nmr_peaks': h_nmr_peaks,
                'c_nmr_peaks': c_nmr_peaks
            }
            
            # 添加molecular_formula（如果存在）
            if 'molecular_formula' in item:
                processed_item['molecular_formula'] = item['molecular_formula']
            
            # 裁剪峰值数量以避免内存问题
            if len(h_nmr_peaks) > self.max_peaks:
                processed_item['h_nmr_peaks'] = h_nmr_peaks[:self.max_peaks]
                logger.debug(f"样本 {idx} H-NMR 峰值数量 ({len(h_nmr_peaks)}) 超过最大限制 ({self.max_peaks})，已裁剪")
            
            if len(c_nmr_peaks) > self.max_peaks:
                processed_item['c_nmr_peaks'] = c_nmr_peaks[:self.max_peaks]
                logger.debug(f"样本 {idx} C-NMR 峰值数量 ({len(c_nmr_peaks)}) 超过最大限制 ({self.max_peaks})，已裁剪")
            
            return processed_item
            
        except Exception as e:
            logger.error(f"获取样本 {idx} 时出错: {str(e)}")
            return None


__all__ = ["MergedDataset"]