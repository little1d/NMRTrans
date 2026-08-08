<div align="center">

# 🤓 [KDD 2026] NMRTrans: Structure Elucidation from Experimental NMR Spectra via Set Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2602.10158-b31b1b.svg)](https://arxiv.org/abs/2602.10158)
[![ACM Digital Library](https://img.shields.io/badge/ACM%20Digital%20Library-10.1145%2F3770855.3818935-0085CA?logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3770855.3818935)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Collection-yellow?logo=huggingface)](https://huggingface.co/collections/little1d/nmrtrans)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn/@Harrison/NMRTrans/overview)

</div>

NMRTrans is a transformer-based framework that performs structure elucidation from experimental NMR spectra. By leveraging Set Transformers with Induced Set Attention Blocks (ISAB) and Pooling by Multihead Attention (PMA), NMRTrans encodes unordered NMR peak sets into modality-specific representations. The framework fuses these representations with optional molecular formula constraints and employs a T5 decoder for autoregressive SMILES generation, effectively handling the permutation-invariant nature of spectral data while maintaining chemical validity.

<div align="center">
  <img src="assets/main.png" alt="NMRTrans Framework" width="90%">
</div>

## 📢 Latest News
- **2026.7.2**: Pretrained checkpoints released and the codebase has been refactored.
- **2026.5.17**: NMRTrans was accepted to the KDD 2026 AI4S Track!
- **2026.3.6**： 🚀 Release training, inference code & datasets.
- **2026.2.10**： 📄 Our paper is now available on [arXiv](https://arxiv.org/pdf/2602.10158).

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/little1d/NMRTrans.git
cd NMRTrans
```

Create a Python 3.10 environment and install dependencies with `uv`:

```bash
conda create -n nmrtrans python=3.10 -y
conda activate nmrtrans

pip install uv
uv pip install -e .
```

## ⚙️ Configuration

NMRTrans uses a single YAML entry point for both training and inference:

```bash
cp configs/config.yaml configs/local.yaml
# Edit configs/local.yaml for local paths, checkpoints, data paths, and runtime settings.
```

Paths in YAML files are resolved relative to the project root. At least one NMR modality must be enabled with `USE_C_NMR` or `USE_H_NMR`; molecular formula guidance is optional through `USE_FORMULA_GUIDANCE`.

`src/config_local.py` is still supported for compatibility only when no YAML file is passed. When `--config_path` is provided, configuration is resolved from project defaults plus that YAML file, without inheriting values from `config_local.py`. To use local Python overrides instead of YAML, copy the template and run `python src/train.py` without `--config_path`:

```bash
cp src/config_local.py.example src/config_local.py
```

## 🔬 Inference

Pretrained checkpoints are available from the [NMRTrans Hugging Face collection](https://huggingface.co/collections/little1d/nmrtrans). Download the checkpoint that matches the input features you want to use.

For the C-NMR + H-NMR + Formula baseline:

```bash
mkdir -p checkpoints/pretrained
huggingface-cli download little1d/C-H-Formula nmrtrans-c-h-nmr-formula.ckpt --local-dir checkpoints/pretrained

python src/test.py \
  --config_path configs/experiment_c_h_formula.yaml \
  --ckpt_path checkpoints/pretrained/nmrtrans-c-h-nmr-formula.ckpt \
  --features c_nmr,h_nmr,formula \
  --output_dir checkpoints/eval_c_h_formula
```

Other released checkpoints can be downloaded in the same way by substituting the Hugging Face repository and checkpoint filename:

| Input features | Hugging Face repo | Checkpoint file | `--features` |
| --- | --- | --- | --- |
| C-NMR | `little1d/C` | `nmrtrans-c-nmr.ckpt` | `c_nmr` |
| H-NMR | `little1d/H` | `nmrtrans-h-nmr.ckpt` | `h_nmr` |
| C-NMR + H-NMR | `little1d/C-H` | `nmrtrans-c-h-nmr.ckpt` | `c_nmr,h_nmr` |
| C-NMR + Formula | `little1d/C-Formula` | `nmrtrans-c-nmr-formula.ckpt` | `c_nmr,formula` |
| H-NMR + Formula | `little1d/H-Formula` | `nmrtrans-h-nmr-formula.ckpt` | `h_nmr,formula` |
| C-NMR + H-NMR + Formula | `little1d/C-H-Formula` | `nmrtrans-c-h-nmr-formula.ckpt` | `c_nmr,h_nmr,formula` |

Template:

```bash
huggingface-cli download <repo_id> <checkpoint_file> --local-dir checkpoints/pretrained

python src/test.py \
  --config_path configs/experiment_<feature_combination>.yaml \
  --ckpt_path checkpoints/pretrained/<checkpoint_file> \
  --features <feature_list> \
  --output_dir checkpoints/<evaluation_name>
```

Use the experiment configuration matching the checkpoint's feature combination,
and make sure its data/cache paths point to the released test split.

## 📊 Results

The table reports greedy top-1 autoregressive decoding on the released
21,298-sample test split. Sequence accuracy compares the generated and target
molecules after RDKit SMILES canonicalization, matching the validation metric;
equivalent SMILES serializations therefore count as correct. Epochs refer to
the selected validation-best checkpoints used for release.

| Input features | Epoch | Sequence acc. | Token acc. | Tanimoto similarity |
| --- | ---: | ---: | ---: | ---: |
| C-NMR | 8519 | 0.1364 | 0.4835 | 0.4694 |
| H-NMR | 7459 | 0.2057 | 0.5768 | 0.5617 |
| C-NMR + H-NMR | 7459 | 0.3497 | 0.6795 | 0.6944 |
| C-NMR + Formula | 9059 | 0.2551 | 0.5826 | 0.6015 |
| H-NMR + Formula | 9579 | 0.3590 | 0.6698 | 0.6904 |
| C-NMR + H-NMR + Formula | 5399 | 0.4337 | 0.7263 | 0.7592 |

> **Notes**
> - All metrics are computed under greedy top-1 autoregressive decoding.
> - These numbers may differ slightly from those reported in the paper because they were re-evaluated with the refactored codebase and released checkpoints.
> - Full metrics, including top-3/5/10 sequence accuracy, are available under `results/release_v1/`.

## 🏋️ Training

Training from scratch requires the T5 backbone and the preprocessed NMRTrans data cache.

Download the T5 backbone:

```bash
mkdir -p models
huggingface-cli download t5-small --local-dir models/t5-small
```

Download the released training, validation, and test splits from the [NMRTrans-Data dataset repository](https://huggingface.co/datasets/little1d/NMRTrans-Data):

```bash
mkdir -p cache
huggingface-cli download little1d/NMRTrans-Data --repo-type dataset --local-dir cache
```

The released cache contains the pre-split train, validation, and test files used by the paper and the released checkpoints. Place them under `NMRTrans/cache` as shown above, then make sure the dataset paths in `configs/local.yaml` point to these files.

Train with the local YAML configuration:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p checkpoints

python src/train.py --config_path configs/local.yaml
```

The default example configuration is designed for 4 GPUs with `BATCH_SIZE=1024`. You can reduce the GPU count and batch size to run on smaller hardware, for example a single GPU with `BATCH_SIZE=128`. When changing the effective batch size, consider tuning related optimization parameters such as `Learning rate` and `ACCUM_GRAD_BATCHES`. These changes may affect convergence speed and final performance.

To train a different input combination, edit `USE_C_NMR`, `USE_H_NMR`, and `USE_FORMULA_GUIDANCE` in `configs/local.yaml`, or use one of the prepared experiment YAML files under `configs/`:

```bash
python src/train.py --config_path configs/experiment_c_h_formula.yaml
```

Resume training from a Lightning checkpoint:

```bash
python src/train.py \
  --config_path configs/local.yaml \
  --ckpt_path checkpoints/path/to/checkpoint.ckpt
```

We have open sourced the full training curves and experiment parameters on
[SwanLab](https://swanlab.cn/@Harrison/NMRTrans/overview) for reproducibility.

<div align="center">
  <img src="assets/swanlab.png" alt="NMRTrans SwanLab Training Curves" width="90%">
</div>

## 🔧 Finetuning

NMRTrans checkpoints can be used as initialization for further training, but finetuning on new data is not always a plug-and-play data replacement. For in-distribution 1D NMR datasets with the same preprocessed format, users can usually start from an existing checkpoint and update the dataset paths in the YAML config.

For new spectral settings or additional modalities, such as 2D NMR or HSQC, the data pipeline and model interface should be adapted consistently. In practice, this may require updating the raw-data parser, the serialized dataset format, `MergedDataset`, the collate function, feature normalization, modality masks, and the corresponding encoder/fusion inputs in the model. After these changes, a pretrained NMRTrans checkpoint can still provide a useful initialization for compatible parts of the architecture, while newly introduced modules may need to be initialized and trained from scratch.

To continue training from a compatible checkpoint:

```bash
python src/train.py \
  --config_path configs/local.yaml \
  --ckpt_path checkpoints/path/to/checkpoint.ckpt
```

## 📝 Citation

If you use NMRTrans in your research, please cite:

```bibtex
@inproceedings{10.1145/3770855.3818935,
author = {Yang, Liujia and Yang, Zhuo and Xie, Jiaqing and Wang, Yubin and Gao, Ben and Wei, Xingjian and Sun, Jiaxing and Wu, Jiang and He, Conghui and Li, Yuqiang and Gu, Qinying},
title = {NMRTrans: Structure Elucidation from Experimental NMR Spectra via Set Transformers},
year = {2026},
isbn = {9798400722592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3770855.3818935},
doi = {10.1145/3770855.3818935},
abstract = {Nuclear Magnetic Resonance (NMR) spectroscopy is fundamental for molecular structure elucidation, yet interpreting spectra at scale remains time-consuming and highly expertise-dependent. While recent spectrum-as-language modeling and retrieval-based methods have shown promise, they rely heavily on large corpora of computed spectra and exhibit notable performance drops when applied to experimental measurements. To address these issues, we build NMRSpec, a large-scale corpus of experimental 1H and 13C NMR spectra mined from chemical literature, and propose NMRTrans, which models spectra as unordered peak sets and aligns the model's inductive bias with the physical nature of NMR. To the best of our knowledge, NMRTrans is the first NMR Transformer trained solely on large-scale experimental spectra and achieves state-of-the-art performance on experimental benchmarks, improving Top-10 Accuracy over the strongest baseline by +17.82 points (61.15\% versus 43.33\%), and underscoring the importance of experimental data and structure-aware architectures for reliable NMR structure elucidation. Code is released at https://github.com/little1d/NMRTrans.},
booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {12621–12632},
numpages = {12},
keywords = {ai for chemistry, ai for spectra, nuclear magnetic resonance, structure elucidation, spectra-to-smiles, experimental spectra},
location = {Republic of Korea},
series = {KDD '26}
}
```

## 📬 Contact

Thank you for your interest in NMRTrans. If you have questions about the algorithm, implementation details, or issues encountered while running the code, please open a GitHub issue so the discussion can help other users as well. You can also reach us by email at yzachary1551@gmail.com.

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
