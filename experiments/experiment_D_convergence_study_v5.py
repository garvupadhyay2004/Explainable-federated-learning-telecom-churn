"""Experiment D: controlled FedAvg convergence study for telecom churn.

SAFE DEFAULT: running this file without --run performs synthetic validation only.
No dataset is loaded and no model is trained unless --run is supplied explicitly.

Place this file beside noniid_equal_size_experiment_export_validate.py and run
`python experiment_D_convergence_study_v5.py` for validation only.
Training, when separately authorized, is invoked with `--run`. Existing output is
never silently overwritten; use `--resume` or `--overwrite` explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import expit

try:
    import noniid_equal_size_experiment_export_validate as exp_b
except ImportError as exc:
    raise ImportError(
        "Keep noniid_equal_size_experiment_export_validate.py in the same "
        "directory as this Experiment D script."
    ) from exc

EXPERIMENT_SCHEMA_VERSION = "D-v5"
PREPROCESSING_SIGNATURE = "drop_duplicates; map_Churn_YesNo; drop_customerID; numeric_TotalCharges; stratified_80_20_then_80_20; train_fit_median_imputer_standard_scaler_onehot_ignore"
ALPHAS = (10.0, 1.0, 0.5, 0.1)
SEEDS = (42, 43, 44, 45, 46)
N_CLIENTS = 3
LEARNING_RATE = 0.08
L2_STRENGTH = 0.001
# Paired design: vary communication rounds at E=3, then vary local epochs at R=50.
CONFIGS = (
    {"config": "R12_E3", "rounds": 12, "local_epochs": 3},
    {"config": "R50_E3", "rounds": 50, "local_epochs": 3},
    {"config": "R100_E3", "rounds": 100, "local_epochs": 3},
    {"config": "R50_E1", "rounds": 50, "local_epochs": 1},
    {"config": "R50_E5", "rounds": 50, "local_epochs": 5},
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid, consistent with logaddexp loss."""
    return expit(np.asarray(z, dtype=np.float64))


def _objective_and_gradient(w, b, X, y, sample_weights):
    """Global weighted logistic objective and its exact full-data gradient."""
    total_weight = float(sample_weights.sum())
    z = X @ w + b
    residual = sample_weights * (_sigmoid(z) - y)
    objective = (
        np.sum(sample_weights * (np.logaddexp(0.0, z) - y * z)) / total_weight
        + 0.5 * L2_STRENGTH * float(w @ w)
    )
    grad_w = (X.T @ residual) / total_weight + L2_STRENGTH * w
    grad_b = float(residual.sum() / total_weight)
    grad_norm = float(np.sqrt(grad_w @ grad_w + grad_b * grad_b))
    return float(objective), grad_w, grad_b, grad_norm


def _validate_one_step_equivalence():
    """Exercise the real FedAvg function and compare E=1 with global GD."""
    rng = np.random.default_rng(20261106)
    X = rng.normal(size=(30, 5))
    y = np.asarray([0] * 19 + [1] * 11, dtype=np.int64)
    sw = compute_sample_weight(class_weight="balanced", y=y).astype(float)
    clients, _ = exp_b.constrained_dirichlet_label_partition(y, 3, 0.5, 17)
    w0 = np.zeros(X.shape[1], dtype=float)
    b0 = 0.0
    _, gw, gb, _ = _objective_and_gradient(w0, b0, X, y.astype(float), sw)
    expected_w = w0 - LEARNING_RATE * gw
    expected_b = b0 - LEARNING_RATE * gb

    # The central reference is only a placeholder for this synthetic one-step test.
    placeholder_train_prob = _sigmoid(X @ w0 + b0)
    actual_w, actual_b, trace = _train_fedavg_with_trace(
        X, y.astype(float), clients, sw, X, y,
        rounds=1, local_epochs=1,
        central_w=w0, central_b=b0,
        central_train_prob=placeholder_train_prob,
    )
    if not (np.allclose(actual_w, expected_w, rtol=1e-11, atol=1e-12)
            and np.isclose(actual_b, expected_b, rtol=1e-11, atol=1e-12)):
        raise AssertionError("Actual E=1 FedAvg function does not match one global full-batch GD step.")
    if len(trace) != 1 or trace[0]["round"] != 1:
        raise AssertionError("Actual FedAvg trace did not record exactly one communication round.")
    print("PASS: actual E=1 FedAvg function equals one global full-batch GD step.")



def _validate_checkpoint_integrity(out: Path, completed_run_keys: set[str]):
    """Ensure every manifest-completed run has its expected persisted CSV rows.

    Extra CSV rows not yet listed in the manifest are allowed: they can occur if
    the process stopped after an atomic CSV write but before manifest update.
    The corresponding run will be safely re-executed and upserted on resume.
    """
    paths = {
        "round": out / "experiment_D_round_history.csv",
        "summary": out / "experiment_D_final_summary.csv",
        "partition": out / "experiment_D_partition_diagnostics.csv",
    }
    frames = {}
    for kind, path in paths.items():
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            if "run_key" not in frame.columns:
                raise ValueError(f"Checkpoint integrity failure: missing run_key column in {path.name}.")
            frames[kind] = frame
        else:
            frames[kind] = pd.DataFrame(columns=["run_key"])

    expected_configs = {cfg["config"]: int(cfg["rounds"]) for cfg in CONFIGS}
    for key in sorted(completed_run_keys):
        if key.startswith("partition|"):
            count = int((frames["partition"]["run_key"] == key).sum())
            if count != 1:
                raise ValueError(
                    f"Checkpoint integrity failure: manifest marks {key!r} complete, "
                    f"but partition diagnostics contain {count} rows (expected 1)."
                )
            continue
        if not key.startswith("alpha=") or "|config=" not in key:
            raise ValueError(f"Checkpoint integrity failure: unrecognized completed run key {key!r}.")
        config_name = key.rsplit("|config=", 1)[-1]
        if config_name not in expected_configs:
            raise ValueError(f"Checkpoint integrity failure: unknown configuration in {key!r}.")
        round_count = int((frames["round"]["run_key"] == key).sum())
        summary_count = int((frames["summary"]["run_key"] == key).sum())
        expected_rounds = expected_configs[config_name]
        if round_count != expected_rounds or summary_count != 1:
            raise ValueError(
                f"Checkpoint integrity failure for {key!r}: found {round_count}/"
                f"{expected_rounds} round rows and {summary_count}/1 summary rows. "
                "Restore the missing output records or use a fresh output directory; "
                "the script will not silently skip this run."
            )


def _validate_checkpoint_integrity_synthetic():
    """Exercise detection of a completed run whose persisted output was deleted."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="experiment_D_v4_integrity_") as tmp:
        out = Path(tmp)
        key = "alpha=1|seed=42|config=R12_E3"
        pd.DataFrame([{"run_key": key, "round": i} for i in range(1, 13)]).to_csv(
            out / "experiment_D_round_history.csv", index=False
        )
        pd.DataFrame([{"run_key": key}]).to_csv(
            out / "experiment_D_final_summary.csv", index=False
        )
        pd.DataFrame([{"run_key": "partition|alpha=1|seed=42"}]).to_csv(
            out / "experiment_D_partition_diagnostics.csv", index=False
        )
        _validate_checkpoint_integrity(out, {key, "partition|alpha=1|seed=42"})
        # Simulate corruption/deletion: manifest still claims completion but one row is gone.
        pd.DataFrame([{"run_key": key, "round": i} for i in range(1, 12)]).to_csv(
            out / "experiment_D_round_history.csv", index=False
        )
        try:
            _validate_checkpoint_integrity(out, {key, "partition|alpha=1|seed=42"})
        except ValueError as exc:
            if "Checkpoint integrity failure" not in str(exc):
                raise
        else:
            raise AssertionError("Checkpoint integrity test failed to detect a missing completed-run row.")
    print("PASS: checkpoint integrity detects missing persisted rows for manifest-completed runs.")

def run_validation():
    """No data access or model training; synthetic and import-level checks only."""
    exp_b.run_validation()
    _validate_one_step_equivalence()
    _validate_checkpoint_integrity_synthetic()
    assert len(CONFIGS) == 5
    assert len(ALPHAS) * len(SEEDS) * len(CONFIGS) == 100
    assert CONFIGS[0]["rounds"] == 12 and CONFIGS[0]["local_epochs"] == 3
    print("PASS: Experiment D paired design contains 100 planned runs (4 alpha × 5 seeds × 5 configs).")
    print("VALIDATION ONLY: no churn data loaded; no model training performed.")



def _resolve_dataset_path(data_path: str | None) -> Path:
    candidates = [data_path, "data/Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError(
        "IBM Telco CSV not found. Place it in data/ or pass --data PATH. "
        "The script will not download data implicitly."
    )


def _load_and_prepare(data_path: str | None):
    source_path = _resolve_dataset_path(data_path)
    print(f"Loading local dataset: {source_path}")
    df = pd.read_csv(source_path)
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' is missing.")
    df = df.drop_duplicates().copy()
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df.pop("Churn").astype(int).to_numpy()
    X = df

    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )
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
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_train_t = X_train_t.toarray() if hasattr(X_train_t, "toarray") else np.asarray(X_train_t)
    X_val_t = X_val_t.toarray() if hasattr(X_val_t, "toarray") else np.asarray(X_val_t)
    return (
        np.asarray(X_train_t, dtype=np.float64), np.asarray(X_val_t, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64), np.asarray(y_val, dtype=np.int64),
    )


def _train_fedavg_with_trace(X, y, clients, sample_weights, X_val, y_val,
                             rounds, local_epochs, central_w, central_b,
                             central_train_prob):
    """Full-batch local GD + weighted FedAvg; record diagnostics every round."""
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    total_weight = float(sample_weights.sum())
    rows = []
    for rnd in range(1, rounds + 1):
        local_models, local_weights = [], []
        for idx in clients:
            Xc, yc, wc = X[idx], y[idx], sample_weights[idx]
            wk = float(wc.sum())
            local_w, local_b = w.copy(), float(b)
            for _ in range(local_epochs):
                residual = wc * (_sigmoid(Xc @ local_w + local_b) - yc)
                grad_w = (Xc.T @ residual) / wk + L2_STRENGTH * local_w
                grad_b = float(residual.sum() / wk)
                local_w -= LEARNING_RATE * grad_w
                local_b -= LEARNING_RATE * grad_b
            local_models.append((local_w, local_b))
            local_weights.append(wk)
        w = sum(model[0] * wk for model, wk in zip(local_models, local_weights)) / total_weight
        b = sum(model[1] * wk for model, wk in zip(local_models, local_weights)) / total_weight

        objective, _, _, grad_norm = _objective_and_gradient(w, b, X, y, sample_weights)
        train_prob = _sigmoid(X @ w + b)
        val_prob = _sigmoid(X_val @ w + b)
        rows.append({
            "round": rnd,
            "global_objective": objective,
            "global_gradient_l2_norm": grad_norm,
            "coefficient_l2_distance_to_central": float(np.linalg.norm(np.r_[w, b] - np.r_[central_w, central_b])),
            "train_probability_rmse_vs_central": float(np.sqrt(np.mean((train_prob - central_train_prob) ** 2))),
            "train_probability_max_abs_diff_vs_central": float(np.max(np.abs(train_prob - central_train_prob))),
            "validation_log_loss": float(log_loss(y_val, val_prob, labels=[0, 1])),
            "validation_roc_auc": float(roc_auc_score(y_val, val_prob)),
        })
    return w, b, rows



def _prepare_output_dir(output_dir: str, resume: bool, overwrite: bool) -> Path:
    """Prepare output directory without requiring not-yet-created checkpoint files."""
    out = Path(output_dir)
    if resume and overwrite:
        raise ValueError("Choose either --resume or --overwrite, not both.")
    if out.exists() and any(out.iterdir()):
        if not resume and not overwrite:
            raise FileExistsError(
                f"Output directory is non-empty: {out}. Use --resume to continue "
                "a compatible run or --overwrite to replace its contents."
            )
        if overwrite:
            import shutil
            shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _upsert_run_rows(path: Path, rows: list[dict], run_key: str):
    """Atomically replace one run's rows, making resume safe after interruption."""
    if not rows:
        return
    new_rows = pd.DataFrame(rows)
    if "run_key" not in new_rows.columns:
        new_rows["run_key"] = run_key
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        if "run_key" not in existing.columns:
            raise ValueError(f"Cannot safely resume legacy output without run_key: {path}")
        existing = existing.loc[existing["run_key"] != run_key]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_manifest(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_run_keys": [], "status": "in_progress", "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION}


def run_experiment(data_path: str | None, output_dir: str, resume: bool = False, overwrite: bool = False):
    out = _prepare_output_dir(output_dir, resume=resume, overwrite=overwrite)
    manifest_path = out / "experiment_D_manifest.json"
    manifest = _load_manifest(manifest_path)
    completed = set(manifest.get("completed_run_keys", []))
    if resume:
        _validate_checkpoint_integrity(out, completed)
    source_path = _resolve_dataset_path(data_path)
    dataset_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    partition_module_path = Path(exp_b.__file__).resolve()
    partition_module_hash = hashlib.sha256(partition_module_path.read_bytes()).hexdigest()
    signature_payload = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "script_sha256": script_hash,
        "partition_module_sha256": partition_module_hash,
        "preprocessing_signature": PREPROCESSING_SIGNATURE,
        "dataset_sha256": dataset_hash,
        "alphas": list(ALPHAS), "seeds": list(SEEDS), "clients": N_CLIENTS,
        "configs": list(CONFIGS), "learning_rate": LEARNING_RATE,
        "l2_strength": L2_STRENGTH,
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if resume and manifest.get("run_signature") not in (None, run_signature):
        raise ValueError("Resume signature mismatch: code, partition module, preprocessing, dataset or experiment configuration differs. Use a new output directory.")
    if resume and manifest.get("run_signature") is None and completed:
        raise ValueError("Cannot safely resume legacy manifest with completed runs but no compatibility signature.")
    manifest["experiment_schema_version"] = EXPERIMENT_SCHEMA_VERSION
    manifest["run_signature"] = run_signature
    manifest["status"] = "in_progress"
    manifest["completed_run_keys"] = sorted(completed)
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    X_train, X_val, y_train, y_val = _load_and_prepare(str(source_path))
    print(f"Split: train={len(y_train)}, validation={len(y_val)}; test set is not used.")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float64)

    central_c = 1.0 / (L2_STRENGTH * float(sample_weights.sum()))
    central = LogisticRegression(
        C=central_c, solver="lbfgs", tol=1e-8, max_iter=5000,
        class_weight="balanced", random_state=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        central.fit(X_train, y_train)
    central_w = np.asarray(central.coef_[0], dtype=np.float64)
    central_b = float(central.intercept_[0])
    central_train_prob = central.predict_proba(X_train)[:, 1]
    central_val_prob = central.predict_proba(X_val)[:, 1]
    central_objective, _, _, central_grad_norm = _objective_and_gradient(
        central_w, central_b, X_train, y_train, sample_weights
    )
    central_summary = {
        "central_objective": central_objective,
        "central_gradient_l2_norm": central_grad_norm,
        "central_validation_log_loss": float(log_loss(y_val, central_val_prob, labels=[0, 1])),
        "central_validation_roc_auc": float(roc_auc_score(y_val, central_val_prob)),
        "central_n_iter": int(np.max(central.n_iter_)),
        "central_hit_max_iter_limit": bool(np.max(central.n_iter_) >= 5000),
        "central_c": central_c,
        "l2_strength": L2_STRENGTH,
        "learning_rate": LEARNING_RATE,
    }
    if caught:
        central_summary["central_fit_warnings"] = [str(w.message) for w in caught]
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    central_summary["central_convergence_warning"] = bool(convergence_warnings)
    central_summary["central_reference_accepted"] = bool(
        not convergence_warnings and central_summary["central_n_iter"] < 1500
        and central_grad_norm <= 1e-4
    )
    if not central_summary["central_reference_accepted"]:
        with (out / "experiment_D_central_reference.json").open("w", encoding="utf-8") as f:
            json.dump(central_summary, f, indent=2)
        raise RuntimeError(
            "Centralized reference failed convergence safeguards: "
            f"warning={bool(convergence_warnings)}, n_iter={central_summary['central_n_iter']}, "
            f"gradient_norm={central_grad_norm:.3e}. Inspect central_reference.json; "
            "do not interpret this model as the optimum."
        )

    round_path = out / "experiment_D_round_history.csv"
    summary_path = out / "experiment_D_final_summary.csv"
    partition_path = out / "experiment_D_partition_diagnostics.csv"
    for alpha in ALPHAS:
        for seed in SEEDS:
            clients, _ = exp_b.constrained_dirichlet_label_partition(
                y_train, N_CLIENTS, alpha, seed
            )
            coefficients = exp_b.validate_partition(y_train, clients, sample_weights)
            heterogeneity = exp_b.weighted_label_heterogeneity(y_train, clients)
            partition_row = {
                "alpha": alpha, "seed": seed,
                "weighted_label_heterogeneity_sd": heterogeneity,
                "client_sizes": ";".join(str(len(c)) for c in clients),
                "aggregation_coefficients": ";".join(f"{v:.12f}" for v in coefficients),
            }
            partition_key = f"partition|alpha={alpha:g}|seed={seed}"
            if partition_key not in completed:
                _upsert_run_rows(partition_path, [{**partition_row, "run_key": partition_key}], partition_key)
                completed.add(partition_key)
                manifest["completed_run_keys"] = sorted(completed)
                manifest["status"] = "in_progress"
                manifest_tmp = manifest_path.with_suffix(".json.tmp")
                manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                os.replace(manifest_tmp, manifest_path)
            for cfg in CONFIGS:
                run_key = f"alpha={alpha:g}|seed={seed}|config={cfg['config']}"
                if run_key in completed:
                    print(f"SKIP completed {run_key}")
                    continue
                w, b, trace = _train_fedavg_with_trace(
                    X_train, y_train, clients, sample_weights, X_val, y_val,
                    cfg["rounds"], cfg["local_epochs"], central_w, central_b,
                    central_train_prob,
                )
                final = trace[-1]
                summary_row = {
                    "alpha": alpha, "seed": seed, **cfg,
                    "final_global_objective": final["global_objective"],
                    "objective_gap_to_central": final["global_objective"] - central_objective,
                    "final_global_gradient_l2_norm": final["global_gradient_l2_norm"],
                    "final_coefficient_l2_distance_to_central": final["coefficient_l2_distance_to_central"],
                    "final_train_probability_rmse_vs_central": final["train_probability_rmse_vs_central"],
                    "final_validation_log_loss": final["validation_log_loss"],
                    "final_validation_roc_auc": final["validation_roc_auc"],
                    "run_key": run_key,
                }
                _upsert_run_rows(round_path, [
                    {"alpha": alpha, "seed": seed, **cfg, **row,
                     "objective_gap_to_central": row["global_objective"] - central_objective,
                     "run_key": run_key}
                    for row in trace
                ], run_key)
                _upsert_run_rows(summary_path, [summary_row], run_key)
                completed.add(run_key)
                manifest["completed_run_keys"] = sorted(completed)
                manifest["status"] = "in_progress"
                manifest_tmp = manifest_path.with_suffix(".json.tmp")
                manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                os.replace(manifest_tmp, manifest_path)
                print(
                    f"alpha={alpha:g} seed={seed} {cfg['config']}: "
                    f"objective={final['global_objective']:.6f}, "
                    f"grad_norm={final['global_gradient_l2_norm']:.3e} [saved]"
                )

    with (out / "experiment_D_central_reference.json").open("w", encoding="utf-8") as f:
        json.dump(central_summary, f, indent=2)
    metadata = {
        "experiment": "D - controlled FedAvg convergence study",
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "preprocessing_signature": PREPROCESSING_SIGNATURE,
        "alphas": list(ALPHAS), "seeds": list(SEEDS), "clients": N_CLIENTS,
        "configs": list(CONFIGS), "planned_fedavg_runs": len(ALPHAS) * len(SEEDS) * len(CONFIGS),
        "learning_rate": LEARNING_RATE, "l2_strength": L2_STRENGTH,
        "test_set_used": False,
        "notes": [
            "Full-batch local gradient descent; local_epochs counts full local-data gradient steps.",
            "E=1 FedAvg should match one global full-batch gradient step per communication round.",
            "Centralized preprocessing and class-weight calculation are retained from Experiment B.",
            "A simulated-client experiment; no differential privacy or secure aggregation is implemented.",
        ],
    }
    with (out / "experiment_D_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    manifest["status"] = "complete"
    manifest["completed_run_keys"] = sorted(completed)
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(f"\nSaved Experiment D outputs to: {out.resolve()}")
    print("No final test-set metrics were computed; the test set remains untouched.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Run safe synthetic checks only (default).")
    parser.add_argument("--run", action="store_true", help="Explicitly load data and execute the 100-model convergence study.")
    parser.add_argument("--data", default=None, help="Path to the local IBM Telco CSV.")
    parser.add_argument("--output-dir", default="experiment_D_v5_outputs")
    parser.add_argument("--resume", action="store_true", help="Resume compatible completed runs in the output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the output directory before training.")
    args = parser.parse_args()
    if args.run:
        if args.resume and args.overwrite:
            parser.error("Choose either --resume or --overwrite, not both.")
        run_validation()
        run_experiment(args.data, args.output_dir, resume=args.resume, overwrite=args.overwrite)
    else:
        run_validation()
        print("Training was NOT executed. Use --run only after a separate explicit decision.")


if __name__ == "__main__":
    main()
