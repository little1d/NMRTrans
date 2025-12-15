# tokenizer.py
import re
import json
import os
from collections import Counter
from typing import Dict, List, Union, Tuple, Optional
import torch
import pickle
import logging
import lz4.frame

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RegexSMILESTokenizer:
    """使用正则表达式进行SMILES分词的Tokenizer，保留指定的特殊token命名"""
    
    def __init__(self, special_tokens=None):
        # SMILES正则表达式模式
        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        
        # 保留指定的特殊token命名
        self.special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<mask>"]
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        
        # 初始化词汇表
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.vocab[self.unk_token] = len(self.special_tokens)
        self.inv_vocab = {i: token for token, i in self.vocab.items()}
        
        # 添加直接访问特殊token ID的属性
        self.pad_token_id = self.vocab[self.pad_token]  # 0
        self.bos_token_id = self.vocab[self.bos_token]  # 1
        self.eos_token_id = self.vocab[self.eos_token]  # 2
        self.mask_token_id = self.vocab[self.mask_token]  # 3
        self.unk_token_id = self.vocab[self.unk_token]  # 4
    
    @classmethod
    def from_file(cls, file_path: str) -> "RegexSMILESTokenizer":
        """从JSON文件加载词汇表"""
        tokenizer = cls()
        with open(file_path, 'r') as f:
            data = json.load(f)
            tokenizer.vocab = data["vocab"]
            # 转换inv_vocab中的键为整数
            tokenizer.inv_vocab = {int(k): v for k, v in data["inv_vocab"].items()}
        
        # 重新设置特殊token ID属性
        tokenizer.pad_token_id = tokenizer.vocab["<pad>"]  # 0
        tokenizer.bos_token_id = tokenizer.vocab["<bos>"]  # 1
        tokenizer.eos_token_id = tokenizer.vocab["<eos>"]  # 2
        tokenizer.mask_token_id = tokenizer.vocab["<mask>"]  # 3
        tokenizer.unk_token_id = tokenizer.vocab["<unk>"]  # 4
        
        return tokenizer
    
    def build_vocab(self, smiles_list: List[str], max_vocab_size: int = 30000) -> "RegexSMILESTokenizer":
        """Based on SMILES list to build vocabulary"""
        all_tokens = []
        
        # Tokenize and count
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            all_tokens.extend(tokens)
        
        # Store token frequencies as an attribute
        self.token_counts = Counter(all_tokens)  # ADD THIS LINE
        
        sorted_tokens = sorted(self.token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        current_idx = len(self.special_tokens) + 1
        for token, _ in sorted_tokens:
            if current_idx >= max_vocab_size:
                break
            if token not in self.vocab:
                self.vocab[token] = current_idx
                self.inv_vocab[current_idx] = token
                current_idx += 1
                
        # Reset special token IDs
        self.pad_token_id = self.vocab["<pad>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.mask_token_id = self.vocab["<mask>"]
        self.unk_token_id = self.vocab["<unk>"]
                
        return self
    
    def tokenize(self, smiles: str) -> List[str]:
        """使用正则表达式对SMILES进行分词"""
        tokens = re.findall(self.pattern, smiles)
        return tokens
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """将tokens转换为ID，兼容单个token和token列表"""
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """将ID转换为tokens，兼容单个ID和ID列表"""
        if isinstance(ids, (int, torch.Tensor)):
            return self.inv_vocab.get(int(ids), self.unk_token)
        return [self.inv_vocab.get(idx, self.unk_token) for idx in ids]
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True, 
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        **kwargs
    ) -> List[int]:
        """将SMILES编码为token ID序列"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        if max_length and truncation:
            tokens = tokens[:max_length]
        
        ids = self.convert_tokens_to_ids(tokens)
        
        if max_length and padding:
            padding_length = max_length - len(ids)
            if padding_length > 0:
                ids = ids + [self.pad_token_id] * padding_length
                
        return ids
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """将token ID序列解码回SMILES"""
        tokens = self.convert_ids_to_tokens(token_ids)
        
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self.pad_token, self.bos_token, self.eos_token, self.mask_token]]
        
        smiles = "".join(tokens)
        
        # 清理tokenization空格（如果需要）
        if clean_up_tokenization_spaces:
            smiles = re.sub(r"\s+", "", smiles)
            
        return smiles
    
    def get_vocab(self) -> Dict[str, int]:
        """返回词汇表"""
        return self.vocab
    
    def add_tokens(self, new_tokens: Union[str, List[str]], special_tokens: bool = False) -> int:
        """添加新tokens到词汇表"""
        if not isinstance(new_tokens, list):
            new_tokens = [new_tokens]
        
        added_count = 0
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.vocab) - 1] = token
                added_count += 1
        
        return added_count
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """保存词汇表"""
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, 'w') as f:
            json.dump({
                "vocab": self.vocab,
                "inv_vocab": {str(k): v for k, v in self.inv_vocab.items()}
            }, f, indent=2)
        return (vocab_file,)
    

def load_data(file_path: str) -> list:
    """加载lz4压缩的pkl文件"""
    with lz4.frame.open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 提取所有SMILES字符串
    smiles_list = [item["original_smiles"] for item in data if item and "original_smiles" in item]
    logger.info(f"从 {file_path} 加载了 {len(smiles_list)} 个SMILES")
    return smiles_list

def build_full_vocab(data_dir: str, output_path: str, max_vocab_size: int = 30000) -> None:
    """基于所有数据集构建完整词汇表"""
    # 加载所有数据集
    all_smiles = []
    
    for dataset_type in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{dataset_type}.pkl.lz4")
        if os.path.exists(file_path):
            smiles_list = load_data(file_path)
            all_smiles.extend(smiles_list)
            logger.info(f"已添加 {dataset_type} 数据集，总计 {len(all_smiles)} 个SMILES")
        else:
            logger.warning(f"数据集文件 {file_path} 不存在，跳过")
    
    if not all_smiles:
        raise ValueError("没有找到有效的SMILES数据，请检查数据路径")
    
    # 构建词汇表
    logger.info(f"开始构建词汇表，共 {len(all_smiles)} 个SMILES，目标词汇表大小: {max_vocab_size}")
    tokenizer = RegexSMILESTokenizer()
    tokenizer.build_vocab(all_smiles, max_vocab_size)
    
    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存词汇表
    tokenizer.save_vocabulary(output_path)
    
    # 打印词汇表信息
    logger.info(f"词汇表构建完成，大小: {len(tokenizer.vocab)}")
    logger.info(f"特殊token: {tokenizer.special_tokens}")
    logger.info(f"UNK token: {tokenizer.unk_token}")
    logger.info(f"前10个常见token: {list(tokenizer.vocab.keys())[:10]}")
    logger.info(f"后10个常见token: {list(tokenizer.vocab.keys())[-10:]}")

    # Print all token frequencies
    print("\nToken frequencies (most common first):")
    for token, count in tokenizer.token_counts.most_common():
        print(f"{token}: {count}")
        
    # You can also save this to a file
    with open(os.path.join(output_path, "token_frequencies.txt"), "w") as f:
        for token, count in tokenizer.token_counts.most_common():
            f.write(f"{token}\t{count}\n")

if __name__ == "__main__":
    # 配置路径
    data_dir = "/mnt/shared-storage-user/yangliujia/precomputed_data_peaks/merged/"
    vocab_path = "/mnt/shared-storage-user/yangliujia/spectra_molecule_gen/peaks_to_structure/vocab_regex"
    
    try:
        build_full_vocab(data_dir, vocab_path, max_vocab_size=30000)
        logger.info("词汇表构建成功！")
    except Exception as e:
        logger.error(f"构建词汇表时出错: {str(e)}")
        raise