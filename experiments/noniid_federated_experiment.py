"""Non-IID Dirichlet label-skew FedAvg experiment.

Separate script; does not modify existing project scripts or train on import.
Fixed stratified 64/16/20 split; preprocessing fitted on train only.
Three clients; alpha=(10, 1, 0.5, 0.1); partition seeds=(42..46).
FedAvg: 12 rounds, 3 local full-batch epochs, lr=0.08, L2=0.001.
Thresholds selected on validation data only; test used for final metrics.
This simulates label skew, not feature skew or decentralized preprocessing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from train_with_shap import clean_data, load_data, make_preprocessor, sigmoid

MODEL_SEED = 42
PARTITION_SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (10.0, 1.0, 0.5, 0.1)
N_CLIENTS, ROUNDS, LOCAL_EPOCHS = 3, 12, 3
LEARNING_RATE, L2_STRENGTH = 0.08, 0.001
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def select_threshold(y_true, probabilities):
    rows = []
    for threshold in THRESHOLDS:
        pred = (probabilities >= threshold).astype(np.int64)
        rows.append({"threshold": float(threshold), "f1": float(f1_score(y_true, pred, zero_division=0))})
    table = pd.DataFrame(rows)
    best_f1 = table.f1.max()
    tied = table[np.isclose(table.f1, best_f1)].copy()
    best = tied.assign(distance=(tied.threshold - 0.5).abs()).sort_values(["distance", "threshold"]).iloc[0]
    return float(best.threshold), table


def evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def dirichlet_label_partition(y, n_clients, alpha, seed, max_attempts=1000):
    """Class-wise Dirichlet allocation; all rows assigned once, no forced class balance."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if n_clients < 2 or alpha <= 0:
        raise ValueError("n_clients must be >=2 and alpha must be positive")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y must contain both binary classes encoded 0/1")
    class_indices = {c: np.flatnonzero(y == c) for c in (0, 1)}
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        parts = [[] for _ in range(n_clients)]
        for label in (0, 1):
            indices = class_indices[label].copy()
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
            counts = rng.multinomial(len(indices), proportions)
            for client_id, chunk in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                parts[client_id].extend(chunk.tolist())
        if all(parts):
            clients = [np.asarray(p, dtype=np.int64) for p in parts]
            for client in clients:
                rng.shuffle(client)
            combined = np.concatenate(clients)
            if len(combined) != len(y) or len(np.unique(combined)) != len(y) or not np.array_equal(np.sort(combined), np.arange(len(y))):
                raise RuntimeError("Partition lost or duplicated training rows")
            return clients
    raise RuntimeError(f"Could not create non-empty clients after {max_attempts} attempts (alpha={alpha}, seed={seed})")


def validate_partition(y, clients, sample_weights):
    combined = np.concatenate(clients)
    if len(combined) != len(y) or len(np.unique(combined)) != len(y) or not np.array_equal(np.sort(combined), np.arange(len(y))):
        raise ValueError("Client partitions must cover each training row exactly once")
    totals = np.array([sample_weights[idx].sum() for idx in clients], dtype=float)
    coefficients = totals / sample_weights.sum()
    if not np.isclose(coefficients.sum(), 1.0, atol=1e-12):
        raise AssertionError("Aggregation coefficients do not sum to one")
    return coefficients


def global_objective(X, y, sample_weights, weights, bias):
    z = X @ weights + bias
    losses = np.logaddexp(0.0, z) - y * z
    return float(np.sum(sample_weights * losses) / sample_weights.sum() + 0.5 * L2_STRENGTH * np.dot(weights, weights))


def train_fedavg_partitioned(X, y, clients):
    """Same weighted local objective and W_k/W aggregation as existing FedAvg."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    validate_partition(y, clients, sample_weights)
    total_weight = float(sample_weights.sum())
    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    history = []
    for round_number in range(1, ROUNDS + 1):
        local_parameters, client_weight_totals = [], []
        for indices in clients:
            Xc, yc, wc = X[indices], y[indices], sample_weights[indices]
            Wc = float(wc.sum())
            local_w, local_b = weights.copy(), float(bias)
            for _ in range(LOCAL_EPOCHS):
                p = sigmoid(Xc @ local_w + local_b)
                error = wc * (p - yc)
                grad_w = (Xc.T @ error) / Wc + L2_STRENGTH * local_w
                grad_b = float(error.sum() / Wc)
                local_w -= LEARNING_RATE * grad_w
                local_b -= LEARNING_RATE * grad_b
            local_parameters.append((local_w, local_b))
            client_weight_totals.append(Wc)
        weights = sum(w * W for (w, _), W in zip(local_parameters, client_weight_totals)) / total_weight
        bias = sum(b * W for (_, b), W in zip(local_parameters, client_weight_totals)) / total_weight
        history.append({"round": round_number, "global_weighted_objective": global_objective(X, y, sample_weights, weights, bias)})
    return weights, bias, history


def main():
    parser = argparse.ArgumentParser(description="Run non-IID Dirichlet label-skew FedAvg experiments")
    parser.add_argument("--data", default=None, help="Optional local churn CSV path")
    parser.add_argument("--output-dir", default="noniid_experiment_outputs")
    parser.add_argument("--partition-seeds", nargs="+", type=int, default=list(PARTITION_SEEDS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    args = parser.parse_args()
    if not args.partition_seeds or any(a <= 0 for a in args.alphas):
        parser.error("Provide seeds and positive alpha values")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading local dataset...")
    X, y = clean_data(load_data(args.data))
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=MODEL_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.20, random_state=MODEL_SEED, stratify=y_tv)
    y_train, y_val, y_test = (np.asarray(v, dtype=np.int64) for v in (y_train, y_val, y_test))
    print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    prep = make_preprocessor(X_train)
    Xtr = to_dense(prep.fit_transform(X_train)).astype(np.float64)
    Xv = to_dense(prep.transform(X_val)).astype(np.float64)
    Xte = to_dense(prep.transform(X_test)).astype(np.float64)

    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    central_C = 1.0 / (L2_STRENGTH * float(sw.sum()))
    central = LogisticRegression(C=central_C, solver="lbfgs", max_iter=1500, class_weight="balanced", random_state=MODEL_SEED)
    central.fit(Xtr, y_train)
    cpv, cpt = central.predict_proba(Xv)[:, 1], central.predict_proba(Xte)[:, 1]
    c_threshold, c_table = select_threshold(y_val, cpv)
    c_table.to_csv(out / "central_validation_thresholds.csv", index=False)
    c_tuned, c_fixed = evaluate(y_test, cpt, c_threshold), evaluate(y_test, cpt, 0.5)
    print(f"Centralized C={central_C:.8f} | validation threshold={c_threshold:.2f} | test F1={c_tuned['f1']:.6f}")

    metric_rows, client_rows, convergence_rows, assignment_rows = [], [], [], []
    for alpha in args.alphas:
        for seed in args.partition_seeds:
            print(f"\n--- alpha={alpha:g}, partition seed={seed} ---")
            clients = dirichlet_label_partition(y_train, N_CLIENTS, alpha, seed)
            for cid, idx in enumerate(clients):
                yy = y_train[idx]
                assignment_rows.extend({"alpha": alpha, "partition_seed": seed, "client_id": cid, "train_row_index": int(i)} for i in idx)
                client_rows.append({"alpha": alpha, "partition_seed": seed, "client_id": cid, "n_records": len(idx), "class_0_count": int((yy == 0).sum()), "class_1_count": int((yy == 1).sum()), "class_0_proportion": float((yy == 0).mean()), "class_1_proportion": float((yy == 1).mean())})
            coeff = validate_partition(y_train, clients, compute_sample_weight(class_weight="balanced", y=y_train))
            print("Client sizes:", [len(i) for i in clients], "| aggregation coefficients:", np.round(coeff, 6).tolist())
            fw, fb, history = train_fedavg_partitioned(Xtr, y_train, clients)
            convergence_rows.extend({"alpha": alpha, "partition_seed": seed, **h} for h in history)
            fpv, fpt = sigmoid(Xv @ fw + fb), sigmoid(Xte @ fw + fb)
            f_threshold, f_table = select_threshold(y_val, fpv)
            f_table.to_csv(out / f"fedavg_validation_thresholds_alpha_{alpha:g}_seed_{seed}.csv", index=False)
            f_tuned, f_fixed = evaluate(y_test, fpt, f_threshold), evaluate(y_test, fpt, 0.5)
            common = {"alpha": alpha, "partition_seed": seed, "n_clients": N_CLIENTS, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "learning_rate": LEARNING_RATE, "l2_strength": L2_STRENGTH, "final_global_weighted_objective": history[-1]["global_weighted_objective"]}
            for model, threshold_type, threshold, values, objective in [
                ("FedAvg", "validation_tuned", f_threshold, f_tuned, history[-1]["global_weighted_objective"]),
                ("FedAvg", "fixed_0.50", 0.5, f_fixed, history[-1]["global_weighted_objective"]),
                ("Centralized", "validation_tuned", c_threshold, c_tuned, np.nan),
                ("Centralized", "fixed_0.50", 0.5, c_fixed, np.nan),
            ]:
                metric_rows.append({**common, "model": model, "threshold_type": threshold_type, "selected_threshold": threshold, "final_global_weighted_objective": objective, **values})
            print(f"FedAvg threshold={f_threshold:.2f} | test F1 tuned={f_tuned['f1']:.4f} | @0.50={f_fixed['f1']:.4f} | final objective={history[-1]['global_weighted_objective']:.6f}")

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby(["alpha", "model", "threshold_type"], as_index=False).agg(
        n_runs=("f1", "count"), accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"))
    metrics.to_csv(out / "noniid_metrics_by_seed.csv", index=False)
    pd.DataFrame(client_rows).to_csv(out / "client_distribution.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(out / "convergence_history.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(out / "client_assignments.csv", index=False)
    summary.to_csv(out / "noniid_summary.csv", index=False)
    print("\nNON-IID EXPERIMENT COMPLETE\n")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {out.resolve()}")

if __name__ == "__main__":
    main()
