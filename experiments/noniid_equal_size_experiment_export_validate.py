"""Equal-size constrained Dirichlet label-skew FedAvg experiment.

This is a separate experiment. Importing this module does not load data or train.
The partitioner draws class-wise Dirichlet preferences, then solves an exact
integer allocation problem subject to fixed client capacities and preserved
class totals. Training is intentionally not included in validation mode.
The training workflow can export aligned test probabilities and validate
regenerated metrics against a preserved reference run.
"""
from __future__ import annotations

import argparse
import heapq
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

ALPHAS = (10.0, 1.0, 0.5, 0.1)
PARTITION_SEEDS = (42, 43, 44, 45, 46)
N_CLIENTS = 3


def validate_partition(y, clients, sample_weights):
    combined = np.concatenate(clients)
    if (len(combined) != len(y) or len(np.unique(combined)) != len(y)
            or not np.array_equal(np.sort(combined), np.arange(len(y)))):
        raise ValueError("Client partitions must cover every row exactly once")
    totals = np.asarray([sample_weights[idx].sum() for idx in clients], dtype=float)
    coefficients = totals / sample_weights.sum()
    if not np.isclose(coefficients.sum(), 1.0, atol=1e-12):
        raise AssertionError("Aggregation coefficients must sum to one")
    return coefficients


def client_capacities(n_rows: int, n_clients: int) -> np.ndarray:
    """Deterministic near-equal capacities; earlier clients get remainder rows."""
    if n_rows < n_clients or n_clients < 2:
        raise ValueError("Need at least one row per client and at least two clients")
    base, remainder = divmod(int(n_rows), int(n_clients))
    return np.asarray([base + (i < remainder) for i in range(n_clients)], dtype=np.int64)


def _optimal_binary_class_counts(
    class_totals: np.ndarray,
    capacities: np.ndarray,
    preferences: np.ndarray,
) -> np.ndarray:
    """Minimize squared deviation from both class-wise Dirichlet quotas.

    For binary labels, assigning class-1 counts x_k determines class-0 counts
    cap_k-x_k. We minimize sum_k[(x_k-N1*p1_k)^2 +
    (cap_k-x_k-N0*p0_k)^2], subject to integer 0<=x_k<=cap_k and
    sum_k x_k=N1. A min-marginal heap solves this separable convex integer
    resource-allocation problem exactly.
    """
    n0, n1 = map(int, class_totals)
    capacities = np.asarray(capacities, dtype=np.int64)
    p0, p1 = np.asarray(preferences, dtype=float)
    if capacities.sum() != n0 + n1:
        raise ValueError("Client capacities must sum to total class count")
    if len(p0) != len(capacities) or len(p1) != len(capacities):
        raise ValueError("Preference vectors must match number of clients")
    if not (np.isclose(p0.sum(), 1.0) and np.isclose(p1.sum(), 1.0)):
        raise ValueError("Each preference vector must sum to one")

    # Convex separable cost for assigning x class-1 records to client k.
    def cost(k: int, x: int) -> float:
        return (x - n1 * p1[k]) ** 2 + (capacities[k] - x - n0 * p0[k]) ** 2

    # Each next class-1 record is assigned to the client with the lowest
    # incremental cost. Convexity makes these marginal costs nondecreasing.
    allocation = np.zeros(len(capacities), dtype=np.int64)
    heap = []
    for k, cap in enumerate(capacities):
        if cap > 0:
            heapq.heappush(heap, (cost(k, 1) - cost(k, 0), k))
    for _ in range(n1):
        if not heap:
            raise RuntimeError("No remaining client capacity for class-1 allocation")
        _, k = heapq.heappop(heap)
        allocation[k] += 1
        x = int(allocation[k])
        if x < capacities[k]:
            heapq.heappush(heap, (cost(k, x + 1) - cost(k, x), k))
    return allocation


def constrained_dirichlet_label_partition(y, n_clients, alpha, seed):
    """Return equal-size client indices and allocation diagnostics.

    Dirichlet preferences are independently sampled for each class. An exact
    integer constrained allocation then preserves global class counts and
    predetermined client capacities. No client size is allowed to vary.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if n_clients < 2 or alpha <= 0:
        raise ValueError("n_clients must be >=2 and alpha must be positive")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y must contain both binary classes encoded 0/1")

    rng = np.random.default_rng(seed)
    capacities = client_capacities(len(y), n_clients)
    class_indices = {c: np.flatnonzero(y == c) for c in (0, 1)}
    n0, n1 = len(class_indices[0]), len(class_indices[1])
    preferences = np.vstack([
        rng.dirichlet(np.full(n_clients, alpha, dtype=float)),
        rng.dirichlet(np.full(n_clients, alpha, dtype=float)),
    ])

    class1_counts = _optimal_binary_class_counts(
        np.asarray([n0, n1]), capacities, preferences
    )
    allocation = np.vstack([capacities - class1_counts, class1_counts])

    parts = [[] for _ in range(n_clients)]
    for label in (0, 1):
        indices = class_indices[label].copy()
        rng.shuffle(indices)
        start = 0
        for client_id in range(n_clients):
            count = int(allocation[label, client_id])
            parts[client_id].extend(indices[start:start + count].tolist())
            start += count
        if start != len(indices):
            raise AssertionError("Class records were not fully allocated")

    clients = [np.asarray(p, dtype=np.int64) for p in parts]
    for client in clients:
        rng.shuffle(client)

    combined = np.concatenate(clients)
    if (len(combined) != len(y)
            or len(np.unique(combined)) != len(y)
            or not np.array_equal(np.sort(combined), np.arange(len(y)))):
        raise RuntimeError("Partition lost or duplicated training rows")
    if not np.array_equal(np.asarray([len(c) for c in clients]), capacities):
        raise RuntimeError("Equal-size capacity constraint was violated")

    diagnostics = {
        "capacities": capacities,
        "class_allocation": allocation,
        "dirichlet_preferences": preferences,
    }
    return clients, diagnostics


def weighted_label_heterogeneity(y, clients):
    """Weighted population SD of client class-1 proportions."""
    y = np.asarray(y, dtype=np.int64)
    sizes = np.asarray([len(idx) for idx in clients], dtype=float)
    proportions = np.asarray([y[idx].mean() for idx in clients], dtype=float)
    global_rate = float(y.mean())
    return float(np.sqrt(np.sum(sizes * (proportions - global_rate) ** 2) / sizes.sum()))


def _brute_force_optimum(class_totals, capacities, preferences):
    """Small-problem oracle used only to validate the heap optimizer."""
    n0, n1 = map(int, class_totals)
    k_count = len(capacities)
    best_cost, best = float("inf"), None
    ranges = [range(int(cap) + 1) for cap in capacities]
    p0, p1 = preferences
    for candidate in product(*ranges):
        if sum(candidate) != n1:
            continue
        value = sum(
            (candidate[k] - n1 * p1[k]) ** 2
            + (capacities[k] - candidate[k] - n0 * p0[k]) ** 2
            for k in range(k_count)
        )
        if value < best_cost - 1e-12:
            best_cost, best = value, np.asarray(candidate, dtype=np.int64)
    return best_cost, best


def run_validation():
    """Synthetic-only tests. Does not load the churn dataset or train models."""
    # 1) Exact optimality against exhaustive enumeration on small cases.
    rng = np.random.default_rng(2026)
    cases = 0
    for capacities in ([3, 2, 4], [2, 3, 2], [4, 4, 3], [1, 3, 2]):
        capacities = np.asarray(capacities, dtype=np.int64)
        for _ in range(25):
            n1 = int(rng.integers(1, capacities.sum()))
            n0 = int(capacities.sum() - n1)
            pref = np.vstack([rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))])
            actual = _optimal_binary_class_counts([n0, n1], capacities, pref)
            expected_cost, expected = _brute_force_optimum([n0, n1], capacities, pref)
            actual_cost = sum(
                (actual[k] - n1 * pref[1, k]) ** 2
                + (capacities[k] - actual[k] - n0 * pref[0, k]) ** 2
                for k in range(3)
            )
            assert np.array_equal(actual, expected), (actual, expected)
            assert np.isclose(actual_cost, expected_cost, atol=1e-10)
            cases += 1
    print(f"PASS 1: Exact integer optimizer matched exhaustive search in {cases} cases.")

    # 2) Full coverage, no overlap, exact capacities, preserved class counts,
    # deterministic repeatability across all requested alpha/seed combinations.
    y = np.asarray([0] * 90 + [1] * 30, dtype=np.int64)
    sw = compute_sample_weight(class_weight="balanced", y=y).astype(float)
    heterogeneity = {}
    for alpha in ALPHAS:
        vals = []
        for seed in PARTITION_SEEDS:
            clients, diag = constrained_dirichlet_label_partition(y, N_CLIENTS, alpha, seed)
            repeated, diag2 = constrained_dirichlet_label_partition(y, N_CLIENTS, alpha, seed)
            assert all(np.array_equal(a, b) for a, b in zip(clients, repeated))
            assert np.array_equal(diag["class_allocation"], diag2["class_allocation"])
            assert [len(c) for c in clients] == diag["capacities"].tolist()
            assert sum(int((y[c] == 0).sum()) for c in clients) == int((y == 0).sum())
            assert sum(int((y[c] == 1).sum()) for c in clients) == int((y == 1).sum())
            coeff = validate_partition(y, clients, sw)
            assert np.isclose(coeff.sum(), 1.0, atol=1e-12)
            vals.append(weighted_label_heterogeneity(y, clients))
        heterogeneity[alpha] = float(np.mean(vals))
    print("PASS 2: Coverage, non-overlap, class totals, exact sizes, reproducibility and aggregation.")

    # 3) Convex marginal costs must be nondecreasing for each client.
    capacities = np.asarray([7, 8, 9])
    pref = np.vstack([rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))])
    n1, n0 = 11, int(capacities.sum() - 11)
    for k, cap in enumerate(capacities):
        def c(x):
            return (x - n1 * pref[1, k]) ** 2 + (cap - x - n0 * pref[0, k]) ** 2
        marginals = np.asarray([c(x + 1) - c(x) for x in range(int(cap))])
        assert np.all(np.diff(marginals) >= -1e-12)
    print("PASS 3: Per-client allocation costs are discretely convex.")

    print("Synthetic label-heterogeneity means (weighted SD):")
    for alpha in ALPHAS:
        print(f"  alpha={alpha:g}: {heterogeneity[alpha]:.6f}")
    print("NOTE: This is a small synthetic diagnostic, not evidence about the churn dataset.")
    print("ALL PARTITION VALIDATION CHECKS PASSED. No model training was performed.")



# ----------------------- Equal-size FedAvg experiment -----------------------

def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _weighted_objective(params, X, y, sample_weights, l2_strength):
    w, b = params[:-1], params[-1]
    total_weight = float(sample_weights.sum())
    z = X @ w + b
    loss = np.sum(sample_weights * (np.logaddexp(0.0, z) - y * z)) / total_weight
    return float(loss + 0.5 * l2_strength * np.dot(w, w))


def train_fedavg_partitioned(
    X, y, clients, sample_weights, rounds=12, local_epochs=3,
    learning_rate=0.08, l2_strength=0.001
):
    """FedAvg with local weighted-mean objectives and W_k/W aggregation."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    total_weight = float(sample_weights.sum())
    history = []

    for rnd in range(1, rounds + 1):
        local_models, client_weight_totals = [], []
        for idx in clients:
            Xc, yc, wc = X[idx], y[idx], sample_weights[idx]
            wk = float(wc.sum())
            local_w, local_b = weights.copy(), float(bias)

            for _ in range(local_epochs):
                error = wc * (_sigmoid(Xc @ local_w + local_b) - yc)
                grad_w = (Xc.T @ error) / wk + l2_strength * local_w
                grad_b = float(error.sum() / wk)
                local_w -= learning_rate * grad_w
                local_b -= learning_rate * grad_b

            local_models.append((local_w, local_b))
            client_weight_totals.append(wk)

        weights = sum(
            model[0] * wk for model, wk in zip(local_models, client_weight_totals)
        ) / total_weight
        bias = sum(
            model[1] * wk for model, wk in zip(local_models, client_weight_totals)
        ) / total_weight

        if rnd in (1, 3, 6, rounds):
            params = np.concatenate([weights, np.asarray([bias])])
            history.append({
                "round": rnd,
                "global_weighted_objective": _weighted_objective(
                    params, X, y, sample_weights, l2_strength
                ),
            })

    return weights, bias, history


def _metric_row(y_true, probabilities, threshold):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def _find_metric_row(metrics, model, threshold_type, alpha, seed):
    mask = (metrics["model"] == model) & (metrics["threshold_type"] == threshold_type)
    if pd.isna(alpha):
        mask &= metrics["alpha"].isna()
    else:
        mask &= np.isclose(metrics["alpha"].astype(float), float(alpha))
    mask &= metrics["seed"].astype(int) == int(seed)
    matches = metrics.loc[mask]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one metric row for model={model}, alpha={alpha}, "
            f"seed={seed}, threshold_type={threshold_type}; found {len(matches)}."
        )
    return matches.iloc[0]


def validate_prediction_metrics(predictions, metrics, expected_test_rows):
    """Recompute fixed/tuned metrics from exported probabilities and cross-check."""
    if predictions.empty:
        raise AssertionError("Prediction export is empty.")

    expected_models = 1 + len(ALPHAS) * len(PARTITION_SEEDS)
    model_groups = predictions.groupby(["model", "alpha", "seed"], dropna=False)
    if len(model_groups) != expected_models:
        raise AssertionError(
            f"Expected {expected_models} prediction groups, found {len(model_groups)}."
        )

    reference_rows = None
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    for (model, alpha, seed), group in model_groups:
        group = group.sort_values("test_position")
        if len(group) != expected_test_rows:
            raise AssertionError(
                f"{model} alpha={alpha} seed={seed}: expected {expected_test_rows} "
                f"test predictions, found {len(group)}."
            )
        if group["test_position"].duplicated().any():
            raise AssertionError("Duplicate test positions found in a prediction group.")
        if not np.array_equal(group["test_position"].to_numpy(), np.arange(expected_test_rows)):
            raise AssertionError("Test positions are missing or out of sequence.")

        identity = group[["test_position", "test_row_index", "y_true"]].reset_index(drop=True)
        if reference_rows is None:
            reference_rows = identity
        elif not identity.equals(reference_rows):
            raise AssertionError("Test row identities/labels differ between model predictions.")

        y_true = group["y_true"].to_numpy(dtype=int)
        probabilities = group["predicted_probability"].to_numpy(dtype=float)
        selected_threshold = float(group["validation_selected_threshold"].iloc[0])
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise AssertionError("Predicted probabilities must be finite and within [0, 1].")
        if not group["validation_selected_threshold"].nunique() == 1:
            raise AssertionError("A model group contains inconsistent selected thresholds.")

        for threshold_type, threshold in (
            ("fixed_0.50", 0.50),
            ("validation_tuned", selected_threshold),
        ):
            predicted = (probabilities >= threshold).astype(int)
            recomputed = {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, predicted),
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
                "roc_auc": roc_auc_score(y_true, probabilities),
            }
            metric_row = _find_metric_row(metrics, model, threshold_type, alpha, seed)
            for column, value in recomputed.items():
                if not np.isclose(float(metric_row[column]), float(value), rtol=1e-10, atol=1e-12):
                    raise AssertionError(
                        f"Prediction/metric mismatch for {model}, alpha={alpha}, seed={seed}, "
                        f"{threshold_type}, {column}: CSV metric={metric_row[column]}, "
                        f"recomputed={value}."
                    )

    print(
        f"PASS: Exported probabilities reproduce fixed/tuned metrics for "
        f"{expected_models} models across {expected_test_rows} aligned test records."
    )


def validate_against_reference(current_metrics, current_central, reference_output_dir,
                               rtol=1e-7, atol=1e-8):
    """Compare regenerated metrics with the preserved original Experiment B run."""
    reference_dir = Path(reference_output_dir)
    metrics_path = reference_dir / "equal_size_metrics_by_seed.csv"
    central_path = reference_dir / "centralized_baseline.csv"
    missing = [str(path) for path in (metrics_path, central_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Reference validation requires the original Experiment B files: " + ", ".join(missing)
        )

    comparisons = (
        ("all model metrics", current_metrics, pd.read_csv(metrics_path)),
        ("centralized baseline", current_central, pd.read_csv(central_path)),
    )
    key_columns = ["model", "threshold_type", "alpha", "seed"]
    value_columns = [
        "threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
        "validation_selected_threshold",
    ]

    for label, current, reference in comparisons:
        for frame_name, frame in (("current", current), ("reference", reference)):
            missing_columns = set(key_columns + value_columns) - set(frame.columns)
            if missing_columns:
                raise AssertionError(
                    f"{label}: {frame_name} file is missing columns {sorted(missing_columns)}."
                )
            if frame.duplicated(key_columns).any():
                raise AssertionError(f"{label}: duplicate model/threshold keys in {frame_name} data.")

        current = current[key_columns + value_columns].copy()
        reference = reference[key_columns + value_columns].copy()
        merged = current.merge(
            reference, on=key_columns, how="outer", suffixes=("_new", "_original"),
            indicator=True,
        )
        if not (merged["_merge"] == "both").all() or len(current) != len(reference):
            raise AssertionError(
                f"{label}: current and original runs have different configuration/threshold rows."
            )
        max_diffs = {}
        for column in value_columns:
            new_values = merged[f"{column}_new"].to_numpy(dtype=float)
            old_values = merged[f"{column}_original"].to_numpy(dtype=float)
            if not np.allclose(new_values, old_values, rtol=rtol, atol=atol, equal_nan=True):
                differences = np.abs(new_values - old_values)
                finite = differences[np.isfinite(differences)]
                max_diff = float(finite.max()) if finite.size else float("nan")
                raise AssertionError(
                    f"{label}: {column} differs from original run (max absolute difference "
                    f"{max_diff:.3e}; tolerance rtol={rtol:g}, atol={atol:g})."
                )
            differences = np.abs(new_values - old_values)
            finite = differences[np.isfinite(differences)]
            max_diffs[column] = float(finite.max()) if finite.size else 0.0
        max_overall = max(max_diffs.values(), default=0.0)
        print(
            f"PASS: {label} matches original run ({len(merged)} rows; "
            f"maximum absolute numeric difference={max_overall:.3e})."
        )


def _choose_threshold(y_true, probabilities):
    from sklearn.metrics import f1_score
    candidates = np.round(np.arange(0.10, 0.901, 0.01), 2)
    scored = [
        (float(f1_score(y_true, probabilities >= t, zero_division=0)), float(t))
        for t in candidates
    ]
    best_f1 = max(item[0] for item in scored)
    tied = [t for score, t in scored if np.isclose(score, best_f1)]
    return min(tied, key=lambda t: abs(t - 0.5))


def run_training(data_path=None, output_dir="noniid_equal_size_outputs",
                 reference_output_dir="noniid_equal_size_outputs_run1"):
    """Train 20 FedAvg configurations. Called only with --run-training."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import confusion_matrix

    # Reuse the project's current data handling/preprocessing conventions.
    candidates = [data_path, "data/Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    df = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            print(f"Loading local dataset: {candidate}")
            df = pd.read_csv(candidate)
            break
    if df is None:
        raise FileNotFoundError(
            "Dataset not found. Place IBM Telco-Customer-Churn.csv in data/ "
            "or pass --data PATH. This experiment will not download data implicitly."
        )
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' was not found.")
    df = df.drop_duplicates().copy()
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df.pop("Churn").astype(int).to_numpy()
    X = df

    # Exact 64/16/20 split used in the preceding objective-aligned experiment.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )
    print("\nDataset split:")
    print(f"Training records:   {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Final test records: {len(X_test)}")

    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    X_train_t = _to_dense(preprocessor.fit_transform(X_train)).astype(np.float64)
    X_val_t = _to_dense(preprocessor.transform(X_val)).astype(np.float64)
    X_test_t = _to_dense(preprocessor.transform(X_test)).astype(np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    test_row_indices = X_test.index.to_numpy(dtype=np.int64)
    test_positions = np.arange(len(y_test), dtype=np.int64)

    l2_strength = 0.001
    sample_weights = compute_sample_weight("balanced", y_train).astype(np.float64)
    total_weight = float(sample_weights.sum())
    central_c = 1.0 / (l2_strength * total_weight)
    central = LogisticRegression(
        C=central_c, penalty="l2", solver="lbfgs", max_iter=1500,
        class_weight="balanced", random_state=42,
    )
    central.fit(X_train_t, y_train)
    central_val_prob = central.predict_proba(X_val_t)[:, 1]
    central_test_prob = central.predict_proba(X_test_t)[:, 1]
    central_threshold = _choose_threshold(y_val, central_val_prob)

    central_rows = []
    confusion_rows = []
    prediction_rows = []
    for position, source_index, actual, probability in zip(
        test_positions, test_row_indices, y_test, central_test_prob
    ):
        prediction_rows.append({
            "test_position": int(position),
            "test_row_index": int(source_index),
            "y_true": int(actual),
            "model": "Centralized Logistic Regression",
            "alpha": np.nan,
            "seed": 42,
            "predicted_probability": float(probability),
            "validation_selected_threshold": float(central_threshold),
        })
    for threshold_name, threshold in [
        ("fixed_0.50", 0.50), ("validation_tuned", central_threshold)
    ]:
        row = _metric_row(y_test, central_test_prob, threshold)
        row.update({
            "model": "Centralized Logistic Regression",
            "threshold_type": threshold_name,
            "alpha": np.nan, "seed": 42,
            "validation_selected_threshold": central_threshold,
        })
        central_rows.append(row)
        central_pred = (central_test_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, central_pred, labels=[0, 1]).ravel()
        confusion_rows.append({
            "model": "Centralized Logistic Regression",
            "alpha": np.nan, "seed": 42, "threshold_type": threshold_name,
            "threshold": float(threshold), "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        })

    metric_rows, distribution_rows, assignment_rows, convergence_rows = [], [], [], []
    partition_rows = []
    for alpha in ALPHAS:
        for seed in PARTITION_SEEDS:
            clients, _diagnostics = constrained_dirichlet_label_partition(
                y_train, N_CLIENTS, alpha, seed
            )
            coefficients = validate_partition(y_train, clients, sample_weights)
            partition_rows.append({
                "alpha": alpha, "seed": seed,
                "weighted_label_heterogeneity_sd": weighted_label_heterogeneity(y_train, clients),
                "client_size_cv": float(np.std([len(c) for c in clients], ddof=0)
                                        / np.mean([len(c) for c in clients])),
            })
            for client_id, idx in enumerate(clients):
                client_churn_rate = float(y_train[idx].mean())
                distribution_rows.append({
                    "alpha": alpha, "seed": seed, "client_id": client_id,
                    "n_records": len(idx),
                    "no_churn": int(np.sum(y_train[idx] == 0)),
                    "churn": int(np.sum(y_train[idx] == 1)),
                    "churn_rate": client_churn_rate,
                    "sample_weight_total": float(sample_weights[idx].sum()),
                    "aggregation_coefficient": float(coefficients[client_id]),
                })
                for row_idx in idx:
                    assignment_rows.append({
                        "alpha": alpha, "seed": seed,
                        "train_row_index": int(row_idx),
                        "client_id": client_id,
                    })

            fw, fb, history = train_fedavg_partitioned(
                X_train_t, y_train, clients, sample_weights,
                rounds=12, local_epochs=3, learning_rate=0.08,
                l2_strength=l2_strength,
            )
            fed_val_prob = _sigmoid(X_val_t @ fw + fb)
            fed_test_prob = _sigmoid(X_test_t @ fw + fb)
            fed_threshold = _choose_threshold(y_val, fed_val_prob)

            for position, source_index, actual, probability in zip(
                test_positions, test_row_indices, y_test, fed_test_prob
            ):
                prediction_rows.append({
                    "test_position": int(position),
                    "test_row_index": int(source_index),
                    "y_true": int(actual),
                    "model": "FedAvg Equal-size Dirichlet",
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "predicted_probability": float(probability),
                    "validation_selected_threshold": float(fed_threshold),
                })

            for item in history:
                convergence_rows.append({
                    "alpha": alpha, "seed": seed, **item
                })

            for threshold_name, threshold in [
                ("fixed_0.50", 0.50), ("validation_tuned", fed_threshold)
            ]:
                row = _metric_row(y_test, fed_test_prob, threshold)
                row.update({
                    "model": "FedAvg Equal-size Dirichlet",
                    "threshold_type": threshold_name,
                    "alpha": alpha, "seed": seed,
                    "validation_selected_threshold": fed_threshold,
                })
                metric_rows.append(row)
                pred = (fed_test_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
                confusion_rows.append({
                    "model": "FedAvg Equal-size Dirichlet",
                    "alpha": alpha, "seed": seed, "threshold_type": threshold_name,
                    "threshold": float(threshold), "TN": int(tn), "FP": int(fp),
                    "FN": int(fn), "TP": int(tp),
                })

            print(
                f"alpha={alpha:g}, seed={seed}: "
                f"tuned threshold={fed_threshold:.2f}, "
                f"test F1={_metric_row(y_test, fed_test_prob, fed_threshold)['f1']:.4f}"
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(central_rows + metric_rows)
    central_metrics = pd.DataFrame(central_rows)
    predictions = pd.DataFrame(prediction_rows)

    # Validate exported predictions against freshly computed metrics, then compare
    # regenerated metrics with the untouched original Experiment B output.
    validate_prediction_metrics(predictions, metrics, expected_test_rows=len(y_test))
    validate_against_reference(metrics, central_metrics, reference_output_dir)

    metrics.to_csv(out / "equal_size_metrics_by_seed.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(out / "equal_size_client_distribution.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(out / "equal_size_client_assignments.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(out / "equal_size_convergence_history.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(out / "equal_size_partition_diagnostics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "equal_size_test_confusion_matrices.csv", index=False)
    predictions.to_csv(out / "equal_size_test_predictions.csv", index=False)

    fed = metrics[metrics["model"] == "FedAvg Equal-size Dirichlet"]
    summary = fed.groupby(["alpha", "threshold_type"]).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        accuracy_mean=("accuracy", "mean"), roc_auc_mean=("roc_auc", "mean"),
        threshold_mean=("validation_selected_threshold", "mean"),
        threshold_std=("validation_selected_threshold", "std"),
    ).reset_index()
    summary.to_csv(out / "equal_size_summary.csv", index=False)
    pd.DataFrame(central_rows).to_csv(out / "centralized_baseline.csv", index=False)

    print("\nCENTRALIZED BASELINE")
    print(pd.DataFrame(central_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nEQUAL-SIZE FEDAVG SUMMARY")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved outputs to: {out.resolve()}")
    print("Centralized baseline is one fit; its metrics are not five independent runs.")


def main():
    parser = argparse.ArgumentParser(
        description="Equal-size constrained Dirichlet partition and FedAvg experiment"
    )
    parser.add_argument("--validate-only", action="store_true",
                        help="Run synthetic partition tests only (default)")
    parser.add_argument("--run-training", action="store_true",
                        help="Explicitly run the 20-model churn experiment")
    parser.add_argument("--data", default=None, help="Path to local IBM Telco CSV")
    parser.add_argument("--output-dir", default="noniid_equal_size_outputs")
    parser.add_argument(
        "--reference-output-dir", default="noniid_equal_size_outputs_run1",
        help="Preserved original Experiment B output directory used for metric validation",
    )
    args = parser.parse_args()

    if args.run_training:
        run_validation()
        run_training(data_path=args.data, output_dir=args.output_dir,
                      reference_output_dir=args.reference_output_dir)
    else:
        run_validation()
        print("Training was not run. Use --run-training only when explicitly authorized.")


if __name__ == "__main__":
    main()
