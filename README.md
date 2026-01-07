```
Spectra2Smiles-AR/
├── src/
│   ├── callbacks.py       # 回调函数（包括checkpoint和SwanLab日志）
│   ├── config.py          # 简化的配置
│   ├── config_local.py    # 本地配置
│   ├── data.py            # MergedDataset数据加载
│   ├── model.py           # T5-based AR模型
│   └── train.py           # 训练脚本
└── scripts/
    ├── rjob.sh
    └── start_training.sh
```
