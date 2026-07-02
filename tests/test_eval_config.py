import unittest
from pathlib import Path
from types import SimpleNamespace

from src.config import PROJECT_ROOT
from src.test import merge_evaluation_results, parse_features, resolve_enabled_features, shard_indices


class EvaluationConfigTests(unittest.TestCase):
    def make_model(self, c_encoder=True, h_encoder=True, formula_encoder=True):
        return SimpleNamespace(
            c_encoder=object() if c_encoder else None,
            h_encoder=object() if h_encoder else None,
            formula_encoder=object() if formula_encoder else None,
        )

    def test_parse_features_returns_none_for_empty_input(self):
        self.assertIsNone(parse_features(None))
        self.assertIsNone(parse_features(""))

    def test_parse_features_normalizes_feature_list(self):
        self.assertEqual(parse_features("C_NMR, formula"), {"c_nmr", "formula"})

    def test_resolve_enabled_features_uses_model_when_cli_features_are_absent(self):
        features = resolve_enabled_features(
            requested_features=None,
            model_module=self.make_model(c_encoder=True, h_encoder=False, formula_encoder=True),
        )

        self.assertEqual(features, {"c_nmr", "formula"})

    def test_resolve_enabled_features_allows_cli_override(self):
        features = resolve_enabled_features(
            requested_features={"h_nmr"},
            model_module=self.make_model(c_encoder=True, h_encoder=True, formula_encoder=True),
        )

        self.assertEqual(features, {"h_nmr"})

    def test_resolve_enabled_features_rejects_missing_encoder(self):
        with self.assertRaisesRegex(ValueError, "h_nmr"):
            resolve_enabled_features(
                requested_features={"h_nmr"},
                model_module=self.make_model(c_encoder=True, h_encoder=False, formula_encoder=True),
            )

    def test_rdkit_morgan_fingerprints_do_not_use_deprecated_api(self):
        deprecated_api = "GetMorganFingerprintAsBitVect"
        sources = [
            (PROJECT_ROOT / "src" / "test.py").read_text(encoding="utf-8"),
            (PROJECT_ROOT / "src" / "model.py").read_text(encoding="utf-8"),
        ]

        self.assertFalse(any(deprecated_api in source for source in sources))

    def test_shard_indices_split_dataset_by_rank(self):
        self.assertEqual(shard_indices(total_size=10, num_shards=3, shard_index=0), [0, 3, 6, 9])
        self.assertEqual(shard_indices(total_size=10, num_shards=3, shard_index=1), [1, 4, 7])
        self.assertEqual(shard_indices(total_size=10, num_shards=3, shard_index=2), [2, 5, 8])

    def test_merge_evaluation_results_uses_weighted_counts(self):
        merged = merge_evaluation_results([
            {
                "metrics": {
                    "token_accuracy": 0.5,
                    "sequence_accuracy": 0.25,
                    "valid_smiles_ratio": 0.75,
                    "tanimoto_similarity": 0.2,
                    "similarity_samples": 2,
                    "total_samples": 4,
                    "examples": [{"rank": 0}],
                    "teacher_forcing_token_accuracy": 0.4,
                    "teacher_forcing_sequence_accuracy": 0.1,
                    "topk_sequence_accuracy": {3: 0.5, 5: 0.75},
                },
                "grouped_metrics": {
                    "short_0-20": {
                        "sample_count": 4,
                        "token_acc": 0.5,
                        "seq_acc": 0.25,
                        "valid_ratio": 0.75,
                        "similarity": 0.2,
                    }
                },
                "inference_time": 9.0,
            },
            {
                "metrics": {
                    "token_accuracy": 1.0,
                    "sequence_accuracy": 0.5,
                    "valid_smiles_ratio": 0.25,
                    "tanimoto_similarity": 0.8,
                    "similarity_samples": 6,
                    "total_samples": 6,
                    "examples": [{"rank": 1}],
                    "teacher_forcing_token_accuracy": 0.8,
                    "teacher_forcing_sequence_accuracy": 0.3,
                    "topk_sequence_accuracy": {3: 1.0, 5: 1.0},
                },
                "grouped_metrics": {
                    "short_0-20": {
                        "sample_count": 6,
                        "token_acc": 1.0,
                        "seq_acc": 0.5,
                        "valid_ratio": 0.25,
                        "similarity": 0.8,
                    }
                },
                "inference_time": 7.0,
            },
        ])

        metrics = merged["metrics"]
        self.assertAlmostEqual(metrics["token_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["sequence_accuracy"], 0.4)
        self.assertAlmostEqual(metrics["valid_smiles_ratio"], 0.45)
        self.assertAlmostEqual(metrics["tanimoto_similarity"], 0.65)
        self.assertEqual(metrics["similarity_samples"], 8)
        self.assertEqual(metrics["total_samples"], 10)
        self.assertEqual(metrics["examples"], [{"rank": 0}, {"rank": 1}])
        self.assertAlmostEqual(metrics["teacher_forcing_token_accuracy"], 0.64)
        self.assertAlmostEqual(metrics["teacher_forcing_sequence_accuracy"], 0.22)
        self.assertAlmostEqual(metrics["topk_sequence_accuracy"][3], 0.8)
        self.assertAlmostEqual(metrics["topk_sequence_accuracy"][5], 0.9)
        self.assertAlmostEqual(merged["grouped_metrics"]["short_0-20"]["token_acc"], 0.8)
        self.assertEqual(merged["grouped_metrics"]["short_0-20"]["sample_count"], 10)
        self.assertEqual(merged["inference_time"], 9.0)


if __name__ == "__main__":
    unittest.main()
