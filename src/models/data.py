class NMRPeakDataset(Dataset):
    """处理lz4压缩的pkl格式NMR数据集"""
    def __init__(self, data_path: str):
        """
        Args:
            data_path: lz4压缩的pkl文件路径
        """
        self.data_path = data_path
        self.data = self._load_data()
        
        # 检查数据集中是否包含H-NMR和C-NMR
        self.has_h_nmr = all('h_nmr_peaks' in sample and sample['h_nmr_peaks'] is not None 
                            for sample in self.data if sample)
        self.has_c_nmr = all('c_nmr_peaks' in sample and sample['c_nmr_peaks'] is not None 
                            for sample in self.data if sample)
        
        if not (self.has_h_nmr or self.has_c_nmr):
            raise ValueError("数据集必须包含h_nmr_peaks或c_nmr_peaks")
    
    def _load_data(self):
        """加载lz4压缩的pkl文件"""
        with lz4.frame.open(self.data_path, 'rb') as f:
            return pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 确保返回的样本格式正确
        return {
            "smiles": sample["smiles"],  # 原始ChemBERTa分词后的token IDs
            "original_smiles": sample["original_smiles"],  # 原始SMILES字符串
            "h_nmr_peaks": sample["h_nmr_peaks"] if self.has_h_nmr else None,
            "c_nmr_peaks": sample["c_nmr_peaks"] if self.has_c_nmr else None,
            "molecular_formula": sample["molecular_formula"] if "molecular_formula" in sample else None
        }