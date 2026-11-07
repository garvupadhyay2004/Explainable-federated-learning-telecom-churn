"""Automated tests for the read-only telecom churn evidence loader.

Run from the project root:
    python -m unittest discover -s tests -v

These tests read the canonical experiment CSVs. They do not train models,
write files, or modify the source evidence.
"""

import unittest

import pandas as pd

from data_loader import (
    B_MODEL_ALPHA_DATASETS,
    C_MODEL_ALPHA_DATASETS,
    CENTRALIZED_MODEL,
    FEDERATED_MODEL,
    EvidenceValidationError,
    _validate_model_alpha,
    get_evidence_inventory,
    load_all_evidence,
    validate_experiment_c,
)


class EvidenceLoaderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load_all_evidence()

    def test_all_three_experiments_are_loaded(self):
        self.assertEqual(set(self.evidence), {"B", "C", "D"})
        self.assertEqual(len(self.evidence["B"]), 9)
        self.assertEqual(len(self.evidence["C"]), 9)
        self.assertEqual(len(self.evidence["D"]), 3)

    def test_canonical_inventory_totals(self):
        inventory = get_evidence_inventory(self.evidence)
        self.assertEqual(len(inventory), 21)
        self.assertEqual(int(inventory["rows"].sum()), 135472)
        self.assertEqual(
            inventory.groupby("experiment")["dataset"].count().to_dict(),
            {"B": 9, "C": 9, "D": 3},
        )

    def test_experiment_d_run_and_round_coverage(self):
        summary = self.evidence["D"]["experiment_D_final_summary"]
        history = self.evidence["D"]["experiment_D_round_history"]
        self.assertEqual(len(summary), 100)
        self.assertEqual(len(history), 5240)
        self.assertFalse(summary["run_key"].duplicated().any())
        self.assertFalse(history.duplicated(["run_key", "round"]).any())
        self.assertEqual(set(summary["run_key"]), set(history["run_key"]))

    def test_model_alpha_semantics_in_b_and_c(self):
        for experiment, dataset_names in (
            ("B", B_MODEL_ALPHA_DATASETS),
            ("C", C_MODEL_ALPHA_DATASETS),
        ):
            for dataset_name in dataset_names:
                with self.subTest(experiment=experiment, dataset=dataset_name):
                    frame = self.evidence[experiment][dataset_name]
                    central = frame.loc[frame["model"] == CENTRALIZED_MODEL]
                    fed = frame.loc[frame["model"] == FEDERATED_MODEL]

                    self.assertGreater(len(central), 0)
                    self.assertTrue(central["alpha"].isna().all())

                    # centralized_baseline.csv is intentionally centralized-only.
                    # The other model/alpha datasets contain both model families.
                    if experiment == "B" and dataset_name == "centralized_baseline":
                        self.assertEqual(len(fed), 0)
                        continue

                    self.assertGreater(len(fed), 0)
                    self.assertTrue(
                        pd.to_numeric(fed["alpha"], errors="coerce").notna().all()
                    )
                    self.assertTrue(
                        set(pd.to_numeric(fed["alpha"]).unique()).issubset(
                            {0.1, 0.5, 1.0, 10.0}
                        )
                    )


class ModelAlphaValidationTests(unittest.TestCase):
    def test_rejects_missing_alpha_for_fedavg(self):
        frame = pd.DataFrame(
            {
                "model": [CENTRALIZED_MODEL, FEDERATED_MODEL],
                "alpha": [float("nan"), float("nan")],
            }
        )
        with self.assertRaisesRegex(EvidenceValidationError, "FedAvg records"):
            _validate_model_alpha(frame, "unit_test")

    def test_rejects_nonmissing_alpha_for_centralized_model(self):
        frame = pd.DataFrame(
            {
                "model": [CENTRALIZED_MODEL, FEDERATED_MODEL],
                "alpha": [1.0, 0.5],
            }
        )
        with self.assertRaisesRegex(EvidenceValidationError, "centralized model alpha"):
            _validate_model_alpha(frame, "unit_test")

    def test_rejects_unexpected_model_label(self):
        frame = pd.DataFrame(
            {
                "model": ["Unexpected Model"],
                "alpha": [0.5],
            }
        )
        with self.assertRaisesRegex(EvidenceValidationError, "unexpected model labels"):
            _validate_model_alpha(frame, "unit_test")

    def test_rejects_infinite_alpha(self):
        frame = pd.DataFrame(
            {
                "model": [CENTRALIZED_MODEL, FEDERATED_MODEL],
                "alpha": [float("nan"), float("inf")],
            }
        )
        with self.assertRaisesRegex(EvidenceValidationError, "infinite"):
            _validate_model_alpha(frame, "unit_test")


class ExplainabilityValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load_all_evidence()

    def test_negative_spearman_is_valid(self):
        """Spearman correlation is valid over [-1, 1], not only [0, 1]."""
        copied = {
            experiment: {
                name: frame.copy(deep=True)
                for name, frame in datasets.items()
            }
            for experiment, datasets in self.evidence.items()
        }
        copied["C"]["shap_rank_summary_by_alpha"].loc[
            copied["C"]["shap_rank_summary_by_alpha"].index[0],
            "transformed_spearman_mean",
        ] = -0.25
        copied["C"]["shap_seed_stability_summary"].loc[
            copied["C"]["shap_seed_stability_summary"].index[0],
            "pairwise_spearman_mean",
        ] = -0.25

        # Validation must accept the mathematically valid negative correlation.
        validate_experiment_c(copied["C"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
