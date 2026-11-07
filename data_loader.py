"""Read-only evidence loader and validation for the telecom churn capstone.

This module loads existing experiment outputs. It never trains models,
creates output files, or modifies experimental evidence.
"""

from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

EVIDENCE_DIRECTORIES = {
    "B": PROJECT_ROOT / "noniid_equal_size_outputs_run2",
    "C": PROJECT_ROOT / "explainability_experiment_C_outputs",
    "D": PROJECT_ROOT / "experiment_D_v5_outputs",
}

CENTRALIZED_MODEL = "Centralized Logistic Regression"
FEDERATED_MODEL = "FedAvg Equal-size Dirichlet"

REQUIRED_FILES = {
    "B": {
        "centralized_baseline.csv": [
            "threshold", "accuracy", "precision", "recall", "f1",
            "roc_auc", "model", "threshold_type", "alpha", "seed",
            "validation_selected_threshold",
        ],
        "equal_size_client_assignments.csv": [
            "alpha", "seed", "train_row_index", "client_id",
        ],
        "equal_size_client_distribution.csv": [
            "alpha", "seed", "client_id", "n_records", "no_churn",
            "churn", "churn_rate", "sample_weight_total",
            "aggregation_coefficient",
        ],
        "equal_size_convergence_history.csv": [
            "alpha", "seed", "round", "global_weighted_objective",
        ],
        "equal_size_metrics_by_seed.csv": [
            "threshold", "accuracy", "precision", "recall", "f1",
            "roc_auc", "model", "threshold_type", "alpha", "seed",
            "validation_selected_threshold",
        ],
        "equal_size_partition_diagnostics.csv": [
            "alpha", "seed", "weighted_label_heterogeneity_sd",
            "client_size_cv",
        ],
        "equal_size_summary.csv": [
            "alpha", "threshold_type", "f1_mean", "f1_std",
            "precision_mean", "recall_mean", "accuracy_mean",
            "roc_auc_mean", "threshold_mean", "threshold_std",
        ],
        "equal_size_test_confusion_matrices.csv": [
            "model", "alpha", "seed", "threshold_type", "threshold",
            "TN", "FP", "FN", "TP",
        ],
        "equal_size_test_predictions.csv": [
            "test_position", "test_row_index", "y_true", "model",
            "alpha", "seed", "predicted_probability",
            "validation_selected_threshold",
        ],
    },
    "C": {
        "model_coefficients.csv": [
            "model", "alpha", "seed", "feature", "coefficient",
        ],
        "shap_global_by_model.csv": [
            "model", "alpha", "seed", "feature",
            "mean_absolute_shap_value", "mean_signed_shap_value",
            "coefficient",
        ],
        "shap_global_grouped_by_model.csv": [
            "model", "alpha", "seed", "feature",
            "mean_absolute_grouped_shap_value",
            "mean_signed_grouped_shap_value",
        ],
        "shap_individual_predictions.csv": [
            "test_position", "test_row_index", "y_true", "model",
            "alpha", "seed", "predicted_probability", "raw_log_odds",
            "shap_base_value",
        ],
        "shap_individual_values.csv": [
            "test_position", "test_row_index", "y_true", "model",
            "alpha", "seed", "feature", "transformed_feature_value",
            "shap_value", "absolute_shap_value", "shap_base_value",
            "raw_log_odds",
        ],
        "shap_rank_correlations.csv": [
            "alpha", "seed", "transformed_feature_spearman",
            "transformed_top10_overlap", "grouped_feature_spearman",
            "grouped_top10_overlap",
        ],
        "shap_rank_summary_by_alpha.csv": [
            "alpha", "transformed_spearman_mean",
            "transformed_spearman_std",
            "transformed_top10_overlap_mean",
            "grouped_spearman_mean", "grouped_spearman_std",
            "grouped_top10_overlap_mean",
        ],
        "shap_seed_stability.csv": [
            "alpha", "feature_level", "seed_a", "seed_b",
            "spearman_rank_correlation", "top10_overlap",
        ],
        "shap_seed_stability_summary.csv": [
            "alpha", "feature_level", "pairwise_spearman_mean",
            "pairwise_spearman_std", "pairwise_top10_overlap_mean",
        ],
    },
    "D": {
        "experiment_D_final_summary.csv": [
            "alpha", "seed", "config", "rounds", "local_epochs",
            "final_global_objective", "objective_gap_to_central",
            "final_global_gradient_l2_norm",
            "final_coefficient_l2_distance_to_central",
            "final_train_probability_rmse_vs_central",
            "final_validation_log_loss", "final_validation_roc_auc",
            "run_key",
        ],
        "experiment_D_partition_diagnostics.csv": [
            "alpha", "seed", "weighted_label_heterogeneity_sd",
            "client_sizes", "aggregation_coefficients", "run_key",
        ],
        "experiment_D_round_history.csv": [
            "alpha", "seed", "config", "rounds", "local_epochs",
            "round", "global_objective", "global_gradient_l2_norm",
            "coefficient_l2_distance_to_central",
            "train_probability_rmse_vs_central",
            "train_probability_max_abs_diff_vs_central",
            "validation_log_loss", "validation_roc_auc",
            "objective_gap_to_central", "run_key",
        ],
    },
}

# Only these B datasets contain both centralized and federated model rows,
# or are the centralized-only baseline. In them, alpha is not applicable
# for centralized records but is mandatory for every FedAvg record.
B_MODEL_ALPHA_DATASETS = {
    "centralized_baseline",
    "equal_size_metrics_by_seed",
    "equal_size_test_confusion_matrices",
    "equal_size_test_predictions",
}

# Experiment C contains centralized reference rows, where alpha is not
# applicable. FedAvg rows must still have a finite alpha.
C_MODEL_ALPHA_DATASETS = {
    "model_coefficients",
    "shap_global_by_model",
    "shap_global_grouped_by_model",
    "shap_individual_predictions",
    "shap_individual_values",
}


class EvidenceValidationError(Exception):
    """Raised when experimental evidence is missing or invalid."""


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list,
    source: str,
) -> None:
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise EvidenceValidationError(
            f"{source}: missing required columns: {missing}"
        )


def _validate_numeric_values(
    dataframe: pd.DataFrame,
    source: str,
    allow_nan_columns: Optional[Set[str]] = None,
) -> None:
    """Reject NaN/infinity except NaN in explicitly approved columns.

    Infinite values are rejected in every column, including columns for which
    NaN is permitted.
    """
    if dataframe.empty:
        raise EvidenceValidationError(
            f"{source}: file contains no records."
        )

    allow_nan_columns = allow_nan_columns or set()
    numeric_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:
        values = dataframe[column].to_numpy(dtype=float)

        if column in allow_nan_columns:
            if np.isinf(values).any():
                raise EvidenceValidationError(
                    f"{source}: column '{column}' contains infinite values."
                )
        elif not np.isfinite(values).all():
            raise EvidenceValidationError(
                f"{source}: column '{column}' contains "
                "NaN or infinite numeric values."
            )


def _validate_probability_columns(
    dataframe: pd.DataFrame,
    columns: list,
    source: str,
) -> None:
    for column in columns:
        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(dataframe[column], errors="coerce")
        if values.isna().any():
            raise EvidenceValidationError(
                f"{source}: '{column}' contains non-numeric values."
            )

        if not values.between(0, 1).all():
            raise EvidenceValidationError(
                f"{source}: '{column}' must be between 0 and 1."
            )


def _validate_model_alpha(
    dataframe: pd.DataFrame,
    source: str,
    centralized_only: bool = False,
) -> None:
    """Require alpha only for FedAvg; centralized alpha is not applicable."""
    if dataframe["model"].isna().any():
        raise EvidenceValidationError(
            f"{source}: model contains missing values."
        )

    models = dataframe["model"]
    alpha = pd.to_numeric(dataframe["alpha"], errors="coerce")

    allowed_models = {CENTRALIZED_MODEL}
    if not centralized_only:
        allowed_models.add(FEDERATED_MODEL)

    unexpected = set(models.unique()) - allowed_models
    if unexpected:
        raise EvidenceValidationError(
            f"{source}: unexpected model labels: {sorted(unexpected)}"
        )

    centralized_mask = models.eq(CENTRALIZED_MODEL)
    federated_mask = models.eq(FEDERATED_MODEL)

    if not centralized_mask.any():
        raise EvidenceValidationError(
            f"{source}: no centralized model records found."
        )

    if not alpha[centralized_mask].isna().all():
        raise EvidenceValidationError(
            f"{source}: centralized model alpha must be missing "
            "because alpha is not applicable."
        )

    if federated_mask.any() and alpha[federated_mask].isna().any():
        raise EvidenceValidationError(
            f"{source}: FedAvg records contain missing or non-numeric alpha."
        )

    if np.isinf(alpha.to_numpy(dtype=float)).any():
        raise EvidenceValidationError(
            f"{source}: alpha contains infinite values."
        )


def _load_experiment(
    experiment: str,
) -> Dict[str, pd.DataFrame]:
    if experiment not in EVIDENCE_DIRECTORIES:
        raise ValueError(
            f"Unknown experiment: {experiment}. "
            f"Expected one of {list(EVIDENCE_DIRECTORIES)}."
        )

    directory = EVIDENCE_DIRECTORIES[experiment]
    if not directory.is_dir():
        raise EvidenceValidationError(
            f"Experiment {experiment} directory not found: {directory}"
        )

    loaded = {}
    errors = []

    for filename, required_columns in REQUIRED_FILES[experiment].items():
        filepath = directory / filename
        if not filepath.is_file():
            errors.append(f"Missing file: {filepath}")
            continue

        try:
            dataframe = pd.read_csv(filepath)
            _validate_columns(dataframe, required_columns, filename)

            dataset_name = filename.removesuffix(".csv")
            allowed_nan_columns = set()

            if (
                (experiment == "B" and dataset_name in B_MODEL_ALPHA_DATASETS)
                or (experiment == "C" and dataset_name in C_MODEL_ALPHA_DATASETS)
            ):
                allowed_nan_columns.add("alpha")

            _validate_numeric_values(
                dataframe,
                filename,
                allow_nan_columns=allowed_nan_columns,
            )

            loaded[dataset_name] = dataframe

        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
            EvidenceValidationError,
            ValueError,
        ) as exc:
            errors.append(f"{filename}: {exc}")

    if errors:
        raise EvidenceValidationError(
            f"Experiment {experiment} validation failed:\n- "
            + "\n- ".join(errors)
        )

    return loaded


def validate_experiment_b(data: Dict[str, pd.DataFrame]) -> None:
    """Validate Experiment B metrics, model/alpha semantics and predictions."""
    for dataset_name in B_MODEL_ALPHA_DATASETS:
        _validate_model_alpha(
            data[dataset_name],
            dataset_name,
            centralized_only=(dataset_name == "centralized_baseline"),
        )

    summary = data["equal_size_summary"]
    valid_threshold_types = {"fixed_0.50", "validation_tuned"}

    if summary["threshold_type"].isna().any() or not set(
        summary["threshold_type"].unique()
    ).issubset(valid_threshold_types):
        raise EvidenceValidationError(
            "Experiment B: unexpected or missing threshold_type."
        )

    for column in [
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    ]:
        if not summary[column].between(0, 1).all():
            raise EvidenceValidationError(
                f"Experiment B: invalid metric values in {column}."
            )

    _validate_probability_columns(
        data["equal_size_test_predictions"],
        [
            "y_true",
            "predicted_probability",
            "validation_selected_threshold",
        ],
        "Experiment B predictions",
    )

    predictions = data["equal_size_test_predictions"]
    if not predictions["y_true"].isin([0, 1]).all():
        raise EvidenceValidationError(
            "Experiment B: y_true must contain only binary labels 0 and 1."
        )

    confusion = data["equal_size_test_confusion_matrices"]
    count_columns = ["TN", "FP", "FN", "TP"]

    if (confusion[count_columns] < 0).any().any():
        raise EvidenceValidationError(
            "Experiment B: negative confusion-matrix counts."
        )

    counts = confusion[count_columns].to_numpy(dtype=float)
    if not np.equal(counts, np.floor(counts)).all():
        raise EvidenceValidationError(
            "Experiment B: confusion-matrix counts must be integers."
        )


def validate_experiment_c(data: Dict[str, pd.DataFrame]) -> None:
    """Validate SHAP model/alpha semantics, rank statistics and contributions."""
    for dataset_name in C_MODEL_ALPHA_DATASETS:
        _validate_model_alpha(data[dataset_name], dataset_name)
    rank = data["shap_rank_summary_by_alpha"]

    # Spearman correlation is mathematically bounded by [-1, 1].
    for column in [
        "transformed_spearman_mean",
        "grouped_spearman_mean",
    ]:
        if not rank[column].between(-1, 1).all():
            raise EvidenceValidationError(
                f"Experiment C: invalid value in {column}; expected [-1, 1]."
            )

    for column in [
        "transformed_top10_overlap_mean",
        "grouped_top10_overlap_mean",
    ]:
        if not rank[column].between(0, 1).all():
            raise EvidenceValidationError(
                f"Experiment C: invalid value in {column}; expected [0, 1]."
            )

    stability = data["shap_seed_stability_summary"]

    if not stability["pairwise_spearman_mean"].between(-1, 1).all():
        raise EvidenceValidationError(
            "Experiment C: pairwise_spearman_mean must be between -1 and 1."
        )

    if not stability["pairwise_top10_overlap_mean"].between(0, 1).all():
        raise EvidenceValidationError(
            "Experiment C: pairwise_top10_overlap_mean must be between 0 and 1."
        )

    individual = data["shap_individual_values"]
    if (individual["absolute_shap_value"] < 0).any():
        raise EvidenceValidationError(
            "Experiment C: absolute SHAP values cannot be negative."
        )


def validate_experiment_d(data: Dict[str, pd.DataFrame]) -> None:
    """Validate Experiment D run coverage and convergence-history structure."""
    summary = data["experiment_D_final_summary"]
    history = data["experiment_D_round_history"]

    if len(summary) != 100:
        raise EvidenceValidationError(
            f"Experiment D: expected 100 final records, found {len(summary)}."
        )

    if len(history) != 5240:
        raise EvidenceValidationError(
            f"Experiment D: expected 5240 round records, found {len(history)}."
        )

    if summary["run_key"].duplicated().any():
        raise EvidenceValidationError(
            "Experiment D: duplicate final-summary run keys."
        )

    if history.duplicated(subset=["run_key", "round"]).any():
        raise EvidenceValidationError(
            "Experiment D: duplicate run/round records."
        )

    if (history["round"] < 1).any():
        raise EvidenceValidationError(
            "Experiment D: round numbers must be positive."
        )

    if (history["round"] > history["rounds"]).any():
        raise EvidenceValidationError(
            "Experiment D: round number exceeds configured rounds."
        )

    expected_rounds = history.groupby("run_key")["round"].nunique()
    configured_rounds = history.groupby("run_key")["rounds"].first()

    if not expected_rounds.equals(configured_rounds):
        raise EvidenceValidationError(
            "Experiment D: round-history counts do not match "
            "configured rounds."
        )

    if set(summary["run_key"]) != set(history["run_key"]):
        raise EvidenceValidationError(
            "Experiment D: final summary and round history have different "
            "run keys."
        )

    _validate_probability_columns(
        summary,
        ["final_validation_roc_auc"],
        "Experiment D final summary",
    )
    _validate_probability_columns(
        history,
        ["validation_roc_auc"],
        "Experiment D round history",
    )


def load_all_evidence() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load and validate all canonical B, C and D evidence."""
    result = {
        "B": _load_experiment("B"),
        "C": _load_experiment("C"),
        "D": _load_experiment("D"),
    }

    validate_experiment_b(result["B"])
    validate_experiment_c(result["C"])
    validate_experiment_d(result["D"])

    return result


def get_evidence_inventory(
    evidence: Dict[str, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Return an in-memory inventory of loaded evidence."""
    records = []

    for experiment, files in evidence.items():
        for name, dataframe in files.items():
            records.append(
                {
                    "experiment": experiment,
                    "dataset": name,
                    "rows": len(dataframe),
                    "columns": len(dataframe.columns),
                }
            )

    return pd.DataFrame(
        records,
        columns=["experiment", "dataset", "rows", "columns"],
    )
