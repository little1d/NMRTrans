"""Shared feature extraction and collation utilities for NMRTrans."""

import json
import logging
import re
from collections import defaultdict
from typing import Iterable, Optional, Set

import torch

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_VOCAB = {
    "<unk>": 0,
    "m": 1,
    "d": 2,
    "s": 3,
    "dd": 4,
    "t": 5,
    "ddd": 6,
    "q": 7,
    "dt": 8,
    "td": 9,
    "br": 10,
    "ddt": 11,
    "dq": 12,
    "tt": 13,
    "quint": 14,
    "dddd": 15,
    "qd": 16,
    "sept": 17,
    "ddp": 18,
    "ddq": 19,
    "bd": 20,
    "dqd": 21,
}

H_NMR_FEATURE_DIM = 10


def pad_peak_sequences(peak_sequences, max_peaks):
    """Pad variable-length peak tensors and return mask where 1=valid and 0=pad."""
    batch_size = len(peak_sequences)
    padded = torch.zeros(batch_size, max_peaks, 1)
    mask = torch.zeros(batch_size, max_peaks, dtype=torch.long)

    for i, peaks in enumerate(peak_sequences):
        num_peaks = min(len(peaks), max_peaks)
        if num_peaks > 0:
            padded[i, :num_peaks] = peaks[:num_peaks]
            mask[i, :num_peaks] = 1

    return padded, mask


def parse_chemical_formula(formula: str) -> dict:
    """Parse a chemical formula string to atom counts."""
    if not formula or formula.strip() == "":
        return {}

    matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip())
    atom_counts = defaultdict(int)
    for atom, count in matches:
        atom_counts[atom] += int(count) if count else 1

    return dict(atom_counts)


def parse_chemical_formula_to_vector(formula: str, atom_mapping: dict) -> torch.Tensor:
    """Convert a chemical formula to an atom-count vector."""
    vec = torch.zeros(len(atom_mapping), dtype=torch.float)
    if not formula or formula.strip() == "":
        return vec

    for atom, count in parse_chemical_formula(formula).items():
        if atom in atom_mapping:
            vec[atom_mapping[atom]] = float(count)

    return vec


def pad_j_coupling(j_coupling_list, max_j=6):
    """Pad J-coupling values to a fixed length."""
    if not isinstance(j_coupling_list, list):
        j_coupling_list = []

    padded = [0.0] * max_j
    for i, val in enumerate(j_coupling_list[:max_j]):
        try:
            padded[i] = float(val)
        except (TypeError, ValueError):
            padded[i] = 0.0
    return padded


def prepare_h_nmr_features(tokenized_input_str, split_vocab=None):
    """Prepare 1H-NMR 10-dimensional peak features from tokenized_input JSON."""
    split_vocab = split_vocab or DEFAULT_SPLIT_VOCAB
    try:
        tokenized_input = json.loads(tokenized_input_str)
        h_nmr_data = tokenized_input.get("1HNMR", [])

        features = []
        for peak in h_nmr_data:
            if len(peak) < 5:
                continue

            chem_shift = float(peak[0])
            peak_width = float(peak[1])
            split_str = str(peak[2]).strip().lower() if len(peak) > 2 else "<unk>"
            split_idx = split_vocab.get(split_str, split_vocab.get("<unk>", 0))

            integral_str = str(peak[3]).strip() if len(peak) > 3 else "1H"
            integral_value = 1.0
            match = re.search(r"(\d+)(?:H|h)?", integral_str)
            if match:
                integral_value = float(match.group(1))

            j_coupling = peak[4] if len(peak) > 4 and isinstance(peak[4], list) else []
            features.append(
                [chem_shift, peak_width, split_idx, integral_value] + pad_j_coupling(j_coupling)
            )

        return features
    except Exception as exc:
        logger.warning(f"Error preparing H-NMR features: {exc}")
        return []


def enabled_features_from_config(config) -> Set[str]:
    """Build enabled feature names from a TrainingConfig-like object."""
    enabled = set()
    if getattr(config, "USE_C_NMR", True):
        enabled.add("c_nmr")
    if getattr(config, "USE_H_NMR", True):
        enabled.add("h_nmr")
    if getattr(config, "USE_FORMULA_GUIDANCE", True):
        enabled.add("formula")
    return enabled


def _normalize_enabled_features(enabled_features: Optional[Iterable[str]]):
    if enabled_features is None:
        return {"c_nmr", "h_nmr", "formula"}
    return set(enabled_features)


def peaks_collate_fn(
    batch,
    tokenizer,
    config,
    atom_mapping=None,
    enabled_features=None,
    apply_jitter=False,
):
    """Collate NMR peak samples for training and evaluation."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    enabled_features = _normalize_enabled_features(enabled_features)

    original_smiles_list = [item["original_smiles"] for item in batch]
    tokenized_smiles = [
        tokenizer.encode(
            smiles,
            max_length=config.MAX_SMILES_LENGTH,
            add_special_tokens=True,
        )
        for smiles in original_smiles_list
    ]

    max_len = config.MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS
    pad_id = tokenizer.vocab["<pad>"]
    smiles_tensor = torch.tensor(
        [tokens + [pad_id] * (max_len - len(tokens)) for tokens in tokenized_smiles],
        dtype=torch.long,
    )

    spectra_data = {}

    if "c_nmr" in enabled_features and any("c_nmr_peaks" in item for item in batch):
        c_peaks_list = []
        for item in batch:
            c_peaks = item.get("c_nmr_peaks", [])
            if c_peaks and len(c_peaks) > 0:
                c_peaks_tensor = torch.tensor(c_peaks, dtype=torch.float)

                if apply_jitter and getattr(config, "NMR_JITTER_RANGE_C", 0) > 0:
                    rounded_peaks = torch.round(c_peaks_tensor * 10) / 10.0
                    jitter = {
                        ppm_val: torch.empty(1)
                        .uniform_(-config.NMR_JITTER_RANGE_C, config.NMR_JITTER_RANGE_C)
                        .item()
                        for ppm_val in torch.unique(rounded_peaks).tolist()
                    }
                    c_peaks_tensor = torch.tensor(
                        [
                            original_val + jitter[ppm_val]
                            for original_val, ppm_val in zip(
                                c_peaks_tensor.tolist(), rounded_peaks.tolist()
                            )
                        ],
                        dtype=torch.float,
                    )

                c_peaks_tensor = torch.clamp(c_peaks_tensor.unsqueeze(-1) / 220.0, 0.0, 1.0)
                c_peaks_list.append(c_peaks_tensor)
            else:
                c_peaks_list.append(torch.zeros((0, 1)))

        c_peaks_padded, c_mask = pad_peak_sequences(c_peaks_list, config.MAX_PEAKS)
        spectra_data["c_nmr_peaks"] = c_peaks_padded
        spectra_data["c_nmr_mask"] = c_mask

    if "h_nmr" in enabled_features and any("tokenized_input" in item for item in batch):
        h_features_list = [
            prepare_h_nmr_features(item.get("tokenized_input", ""), DEFAULT_SPLIT_VOCAB)
            for item in batch
        ]
        max_h_peaks = min(max((len(features) for features in h_features_list), default=0), config.MAX_PEAKS)
        h_features_tensor = torch.zeros(len(batch), max_h_peaks, H_NMR_FEATURE_DIM)
        h_mask = torch.zeros(len(batch), max_h_peaks, dtype=torch.long)

        for i, features in enumerate(h_features_list):
            num_peaks = min(len(features), max_h_peaks)
            if num_peaks > 0:
                h_features_tensor[i, :num_peaks] = torch.tensor(
                    features[:num_peaks], dtype=torch.float
                )
                h_mask[i, :num_peaks] = 1

        spectra_data["h_nmr_features"] = h_features_tensor
        spectra_data["h_nmr_mask"] = h_mask

    if "formula" in enabled_features and getattr(config, "USE_FORMULA_GUIDANCE", True) and atom_mapping:
        formula_strings = [item.get("molecular_formula", "") or "" for item in batch]
        formula_tensor = torch.stack(
            [
                parse_chemical_formula_to_vector(formula, atom_mapping)
                for formula in formula_strings
            ]
        )
        spectra_data["formula_vector"] = formula_tensor
        spectra_data["formula_strings"] = formula_strings

    return {
        "smiles": smiles_tensor,
        "original_smiles": original_smiles_list,
        **spectra_data,
    }


__all__ = [
    "DEFAULT_SPLIT_VOCAB",
    "H_NMR_FEATURE_DIM",
    "enabled_features_from_config",
    "pad_j_coupling",
    "pad_peak_sequences",
    "parse_chemical_formula",
    "parse_chemical_formula_to_vector",
    "peaks_collate_fn",
    "prepare_h_nmr_features",
]
