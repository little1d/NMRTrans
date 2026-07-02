import json
import unittest
from types import SimpleNamespace

import torch

from src.features import (
    DEFAULT_SPLIT_VOCAB,
    peaks_collate_fn,
    prepare_h_nmr_features,
    parse_chemical_formula,
    parse_chemical_formula_to_vector,
)


class FakeTokenizer:
    vocab = {"<pad>": 0}

    def encode(self, smiles, max_length, add_special_tokens=True):
        token_ids = [1]
        token_ids.extend(range(10, 10 + min(len(smiles), max_length - 2)))
        token_ids.append(2)
        return token_ids[: max_length + 2]


def make_config():
    return SimpleNamespace(
        ALL_ATOMS=["B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si"],
        USE_FORMULA_GUIDANCE=True,
        MAX_PEAKS=4,
        MAX_SMILES_LENGTH=6,
        MAX_SMILES_LENGTH_WITH_SPECIAL_TOKENS=8,
        NMR_JITTER_RANGE_C=0.0,
    )


class FeatureProcessingTests(unittest.TestCase):
    def test_formula_parser_and_vector_handle_multichar_atoms(self):
        atoms = parse_chemical_formula("C15H20BrNO3")
        self.assertEqual(atoms, {"C": 15, "H": 20, "Br": 1, "N": 1, "O": 3})

        atom_mapping = {atom: idx for idx, atom in enumerate(make_config().ALL_ATOMS)}
        vector = parse_chemical_formula_to_vector("C15H20BrNO3", atom_mapping)

        self.assertEqual(vector[atom_mapping["Br"]].item(), 1.0)
        self.assertEqual(vector[atom_mapping["C"]].item(), 15.0)
        self.assertEqual(vector[atom_mapping["H"]].item(), 20.0)
        self.assertEqual(vector[atom_mapping["N"]].item(), 1.0)
        self.assertEqual(vector[atom_mapping["O"]].item(), 3.0)

    def test_h_nmr_features_are_ten_dimensional(self):
        tokenized_input = json.dumps(
            {
                "1HNMR": [
                    [7.21, 0.0, "s", "1H", []],
                    [4.35, 0.03, "dd", "2H", [10.4, 8.8]],
                ]
            }
        )

        features = prepare_h_nmr_features(tokenized_input, DEFAULT_SPLIT_VOCAB)

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0], [7.21, 0.0, DEFAULT_SPLIT_VOCAB["s"], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(features[1], [4.35, 0.03, DEFAULT_SPLIT_VOCAB["dd"], 2.0, 10.4, 8.8, 0.0, 0.0, 0.0, 0.0])

    def test_collate_builds_all_feature_tensors(self):
        config = make_config()
        atom_mapping = {atom: idx for idx, atom in enumerate(config.ALL_ATOMS)}
        batch = [
            {
                "original_smiles": "CCO",
                "tokenized_input": json.dumps(
                    {
                        "1HNMR": [[7.21, 0.0, "s", "1H", []]],
                        "13CNMR": [150.0, 20.0],
                    }
                ),
                "c_nmr_peaks": [150.0, 20.0],
                "molecular_formula": "C2H6O",
            },
            {
                "original_smiles": "CN",
                "tokenized_input": json.dumps(
                    {
                        "1HNMR": [[3.0, 0.1, "m", "3H", [7.2]]],
                        "13CNMR": [90.0],
                    }
                ),
                "c_nmr_peaks": [90.0],
                "molecular_formula": "CH5N",
            },
        ]

        collated = peaks_collate_fn(
            batch,
            tokenizer=FakeTokenizer(),
            config=config,
            atom_mapping=atom_mapping,
            enabled_features={"c_nmr", "h_nmr", "formula"},
        )

        self.assertEqual(tuple(collated["smiles"].shape), (2, 8))
        self.assertEqual(tuple(collated["c_nmr_peaks"].shape), (2, 4, 1))
        self.assertEqual(tuple(collated["c_nmr_mask"].shape), (2, 4))
        self.assertEqual(tuple(collated["h_nmr_features"].shape), (2, 1, 10))
        self.assertEqual(tuple(collated["h_nmr_mask"].shape), (2, 1))
        self.assertEqual(tuple(collated["formula_vector"].shape), (2, 12))
        self.assertEqual(collated["formula_strings"], ["C2H6O", "CH5N"])
        self.assertTrue(torch.allclose(collated["c_nmr_peaks"][0, :2, 0], torch.tensor([150.0 / 220.0, 20.0 / 220.0])))

    def test_collate_respects_enabled_features(self):
        config = make_config()
        atom_mapping = {atom: idx for idx, atom in enumerate(config.ALL_ATOMS)}
        batch = [
            {
                "original_smiles": "CCO",
                "tokenized_input": json.dumps({"1HNMR": [[7.21, 0.0, "s", "1H", []]]}),
                "c_nmr_peaks": [150.0],
                "molecular_formula": "C2H6O",
            }
        ]

        collated = peaks_collate_fn(
            batch,
            tokenizer=FakeTokenizer(),
            config=config,
            atom_mapping=atom_mapping,
            enabled_features={"h_nmr"},
        )

        self.assertIn("h_nmr_features", collated)
        self.assertNotIn("c_nmr_peaks", collated)
        self.assertNotIn("formula_vector", collated)


if __name__ == "__main__":
    unittest.main()
