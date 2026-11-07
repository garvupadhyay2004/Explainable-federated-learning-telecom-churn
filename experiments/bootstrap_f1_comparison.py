"""
Paired bootstrap comparison for validation-tuned centralized and FedAvg models.

The models are trained once. Thresholds are selected on validation data only.
The same bootstrap test-row indices are applied to both models in every
replicate. The CI is conditional on the fitted models and selected thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from train_with_shap import (
    clean_data,
    load_data,
    make_preprocessor,
    sigmoid,
    train_fedavg,
)

MODEL_SEED = 42
L2_STRENGTH = 0.001
N_CLIENTS = 3
ROUNDS = 12
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.08
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def select_threshold(y_true, probabilities):
    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "f1": f1_score(y_true, predictions, zero_division=0),
        })

    table = pd.DataFrame(rows)
    best_f1 = table["f1"].max()
    tied = table[np.isclose(table["f1"], best_f1)].copy()
    best = (
        tied.assign(distance=(tied["threshold"] - 0.5).abs())
        .sort_values("distance")
        .iloc[0]
    )
    return float(best["threshold"]), table


def paired_bootstrap_f1(
    y_true,
    central_predictions,
    fedavg_predictions,
    n_bootstrap=10_000,
    random_state=20261105,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    central_predictions = np.asarray(central_predictions, dtype=np.int64)
    fedavg_predictions = np.asarray(fedavg_predictions, dtype=np.int64)

    if not (len(y_true) == len(central_predictions) == len(fedavg_predictions)):
        raise ValueError("Labels and prediction arrays must have equal lengths.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("The original test set must contain both classes.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    rng = np.random.default_rng(random_state)
    n_test = len(y_true)
    central_values = np.empty(n_bootstrap)
    fedavg_values = np.empty(n_bootstrap)
    differences = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Paired resampling: identical row indices for both models.
        idx = rng.integers(0, n_test, size=n_test)
        y_b = y_true[idx]
        central_b = f1_score(y_b, central_predictions[idx], zero_division=0)
        fedavg_b = f1_score(y_b, fedavg_predictions[idx], zero_division=0)

        central_values[b] = central_b
        fedavg_values[b] = fedavg_b
        differences[b] = fedavg_b - central_b

    observed_central = f1_score(
        y_true, central_predictions, zero_division=0
    )
    observed_fedavg = f1_score(
        y_true, fedavg_predictions, zero_division=0
    )
    ci_low, ci_high = np.quantile(differences, [0.025, 0.975])

    summary = {
        "n_test": n_test,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": random_state,
        "centralized_test_f1": observed_central,
        "fedavg_test_f1": observed_fedavg,
        "observed_f1_difference_fedavg_minus_centralized":
            observed_fedavg - observed_central,
        "bootstrap_ci_95_percentile_lower": float(ci_low),
        "bootstrap_ci_95_percentile_upper": float(ci_high),
        "bootstrap_proportion_difference_gt_zero":
            float(np.mean(differences > 0)),
        "bootstrap_proportion_difference_lt_zero":
            float(np.mean(differences < 0)),
        "bootstrap_proportion_difference_equal_zero":
            float(np.mean(differences == 0)),
        "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
    }

    distribution = pd.DataFrame({
        "replicate": np.arange(1, n_bootstrap + 1),
        "centralized_f1": central_values,
        "fedavg_f1": fedavg_values,
        "f1_difference_fedavg_minus_centralized": differences,
    })
    return summary, distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Optional local CSV path.")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument(
        "--output-dir", default="bootstrap_comparison_outputs"
    )
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproduce the existing dataset loading and cleaning.
    df = load_data(args.data)
    X, y = clean_data(df)

    # Reproduce the exact existing stratified 64/16/20 split.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=MODEL_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=MODEL_SEED,
        stratify=y_trainval,
    )

    print("\nDataset split:")
    print(f"Training records:   {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Final test records: {len(X_test)}")

    # Fit preprocessing on training data only.
    preprocessor = make_preprocessor(X_train)
    X_train_t = to_dense(preprocessor.fit_transform(X_train))
    X_val_t = to_dense(preprocessor.transform(X_val))
    X_test_t = to_dense(preprocessor.transform(X_test))

    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    # Centralized model: same weighted objective and L2 mapping.
    sample_weights = compute_sample_weight(
        class_weight="balanced", y=y_train
    )
    total_weight = float(np.sum(sample_weights))
    central_C = 1.0 / (L2_STRENGTH * total_weight)

    central = LogisticRegression(
        C=central_C,
        solver="lbfgs",
        max_iter=1500,
        class_weight="balanced",
        random_state=MODEL_SEED,
    )
    central.fit(X_train_t, y_train)

    central_val_prob = central.predict_proba(X_val_t)[:, 1]
    central_test_prob = central.predict_proba(X_test_t)[:, 1]

    # FedAvg: exact existing configuration.
    fed_weights, fed_bias = train_fedavg(
        X_train_t,
        y_train,
        n_clients=N_CLIENTS,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=MODEL_SEED,
        l2_strength=L2_STRENGTH,
    )
    fed_val_prob = sigmoid(X_val_t @ fed_weights + fed_bias)
    fed_test_prob = sigmoid(X_test_t @ fed_weights + fed_bias)

    # Select thresholds using validation F1 only.
    central_threshold, central_val_table = select_threshold(
        y_val, central_val_prob
    )
    fedavg_threshold, fedavg_val_table = select_threshold(
        y_val, fed_val_prob
    )
    central_val_table.to_csv(
        output_dir / "central_validation_thresholds.csv", index=False
    )
    fedavg_val_table.to_csv(
        output_dir / "fedavg_validation_thresholds.csv", index=False
    )

    print("\nValidation-selected thresholds:")
    print(f"Centralized: {central_threshold:.2f}")
    print(f"FedAvg:      {fedavg_threshold:.2f}")

    central_predictions = (central_test_prob >= central_threshold).astype(int)
    fedavg_predictions = (fed_test_prob >= fedavg_threshold).astype(int)

    summary, distribution = paired_bootstrap_f1(
        y_true=y_test,
        central_predictions=central_predictions,
        fedavg_predictions=fedavg_predictions,
        n_bootstrap=args.n_bootstrap,
        random_state=args.bootstrap_seed,
    )

    summary.update({
        "centralized_threshold": central_threshold,
        "fedavg_threshold": fedavg_threshold,
        "model_seed": MODEL_SEED,
        "l2_strength": L2_STRENGTH,
        "n_clients": N_CLIENTS,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
    })

    pd.DataFrame([summary]).to_csv(
        output_dir / "paired_bootstrap_summary.csv", index=False
    )
    distribution.to_csv(
        output_dir / "paired_bootstrap_distributions.csv", index=False
    )

    print("\nPAIRED BOOTSTRAP RESULTS")
    print(f"Bootstrap replicates: {args.n_bootstrap:,}")
    print(f"Centralized test F1: {summary['centralized_test_f1']:.6f}")
    print(f"FedAvg test F1:      {summary['fedavg_test_f1']:.6f}")
    print(
        "Observed F1 difference (FedAvg - Centralized): "
        f"{summary['observed_f1_difference_fedavg_minus_centralized']:.6f}"
    )
    print(
        "95% percentile CI: "
        f"[{summary['bootstrap_ci_95_percentile_lower']:.6f}, "
        f"{summary['bootstrap_ci_95_percentile_upper']:.6f}]"
    )
    print(
        "Proportion of bootstrap differences > 0: "
        f"{summary['bootstrap_proportion_difference_gt_zero']:.4f}"
    )
    print(f"95% CI includes zero: {summary['ci_includes_zero']}")
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
