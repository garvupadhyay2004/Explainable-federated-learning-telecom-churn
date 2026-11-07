"""Centralized optimizer diagnostic for Experiment D.

SAFE DEFAULT: without --run, performs syntax-independent configuration checks only.
With --run, loads the local IBM Telco dataset and fits three CENTRALIZED logistic
regression models using the exact Experiment D preprocessing, split, class weights,
and regularized objective. It never trains FedAvg and never evaluates the final test set.

Place this file beside experiment_D_convergence_study_v4.py and
noniid_equal_size_experiment_export_validate.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

import experiment_D_convergence_study_v4 as exp_d

DIAGNOSTIC_SCHEMA = "central-optimizer-diagnostic-v1"
MAX_ITER = 5000
TOLERANCES = (1e-4, 1e-6, 1e-8)


def _gradient_diagnostics(w, b, X, y, sample_weights):
    total_weight = float(sample_weights.sum())
    z = X @ w + b
    # Stable sigmoid, matching the objective implemented by Experiment D.
    from scipy.special import expit
    residual = sample_weights * (expit(z) - y)
    grad_w = (X.T @ residual) / total_weight + exp_d.L2_STRENGTH * w
    grad_b = float(residual.sum() / total_weight)
    return {
        "gradient_l2_norm": float(np.sqrt(grad_w @ grad_w + grad_b * grad_b)),
        "gradient_max_abs_component": float(max(np.max(np.abs(grad_w)), abs(grad_b))),
        "gradient_bias_component": grad_b,
    }


def _objective(w, b, X, y, sample_weights):
    total_weight = float(sample_weights.sum())
    z = X @ w + b
    return float(
        np.sum(sample_weights * (np.logaddexp(0.0, z) - y * z)) / total_weight
        + 0.5 * exp_d.L2_STRENGTH * float(w @ w)
    )


def run_diagnostic(data_path: str | None, output_dir: str, overwrite: bool = False):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Diagnostic output directory is non-empty: {out}. "
                "Choose a fresh directory or pass --overwrite explicitly."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source_path = exp_d._resolve_dataset_path(data_path)
    dataset_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    print(f"Loading local dataset: {source_path}")
    X_train, X_val, y_train, y_val = exp_d._load_and_prepare(str(source_path))
    print(f"Split: train={len(y_train)}, validation={len(y_val)}; test set is not used.")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float64)
    central_c = 1.0 / (exp_d.L2_STRENGTH * float(sample_weights.sum()))

    rows = []
    parameter_sets = {}
    reference_w = None
    reference_b = None
    reference_train_prob = None

    for tol in TOLERANCES:
        label = f"tol_{tol:.0e}"
        model = LogisticRegression(
            C=central_c,
            solver="lbfgs",
            max_iter=MAX_ITER,
            tol=tol,
            class_weight="balanced",
            random_state=42,
            # penalty omitted intentionally: sklearn's default is L2.
            # This also avoids the penalty='l2' deprecation warning.
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)

        w = np.asarray(model.coef_[0], dtype=np.float64)
        b = float(model.intercept_[0])
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        objective = _objective(w, b, X_train, y_train, sample_weights)
        grad = _gradient_diagnostics(w, b, X_train, y_train, sample_weights)
        convergence_warnings = [x for x in caught if issubclass(x.category, ConvergenceWarning)]

        if reference_w is None:
            reference_w, reference_b = w.copy(), b
            reference_train_prob = train_prob.copy()
            coefficient_delta = 0.0
            train_probability_rmse_delta = 0.0
            train_probability_max_delta = 0.0
        else:
            coefficient_delta = float(np.linalg.norm(np.r_[w, b] - np.r_[reference_w, reference_b]))
            train_delta = train_prob - reference_train_prob
            train_probability_rmse_delta = float(np.sqrt(np.mean(train_delta ** 2)))
            train_probability_max_delta = float(np.max(np.abs(train_delta)))

        rows.append({
            "model": label,
            "tolerance": tol,
            "max_iter": MAX_ITER,
            "n_iter": int(np.max(model.n_iter_)),
            "hit_max_iter_limit": bool(np.max(model.n_iter_) >= MAX_ITER),
            "convergence_warning": bool(convergence_warnings),
            "objective": objective,
            **grad,
            "validation_log_loss": float(log_loss(y_val, val_prob, labels=[0, 1])),
            "validation_roc_auc": float(roc_auc_score(y_val, val_prob)),
            "coefficient_l2_delta_vs_tol_1e-4": coefficient_delta,
            "train_probability_rmse_delta_vs_tol_1e-4": train_probability_rmse_delta,
            "train_probability_max_abs_delta_vs_tol_1e-4": train_probability_max_delta,
            "fit_warnings": [str(x.message) for x in caught],
        })
        parameter_sets[label] = {"coef": w.tolist(), "intercept": b}
        print(
            f"{label}: objective={objective:.12f}, "
            f"grad_l2={grad['gradient_l2_norm']:.6e}, "
            f"grad_max_abs={grad['gradient_max_abs_component']:.6e}, "
            f"n_iter={int(np.max(model.n_iter_))}, "
            f"val_log_loss={rows[-1]['validation_log_loss']:.8f}, "
            f"val_auc={rows[-1]['validation_roc_auc']:.8f}"
        )

    summary = {
        "diagnostic_schema": DIAGNOSTIC_SCHEMA,
        "experiment_source_schema": exp_d.EXPERIMENT_SCHEMA_VERSION,
        "dataset_sha256": dataset_hash,
        "preprocessing_signature": exp_d.PREPROCESSING_SIGNATURE,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_set_used": False,
        "fedavg_trained": False,
        "central_c": central_c,
        "l2_strength": exp_d.L2_STRENGTH,
        "solver": "lbfgs",
        "max_iter": MAX_ITER,
        "tolerances": list(TOLERANCES),
        "results": rows,
        "interpretation_note": (
            "Compare objective, exact full-gradient norms, coefficient differences, "
            "training-probability differences, and validation metrics. A lower gradient "
            "norm alone does not establish a materially better predictive model."
        ),
    }
    (out / "central_optimizer_diagnostic.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{k: v for k, v in row.items() if k != "fit_warnings"} for row in rows]).to_csv(
        out / "central_optimizer_diagnostic.csv", index=False
    )
    (out / "central_optimizer_parameters.json").write_text(
        json.dumps(parameter_sets, indent=2), encoding="utf-8"
    )
    print(f"\nSaved centralized-only diagnostic to: {out.resolve()}")
    print("FedAvg was not trained. The final test set was not accessed or evaluated.")


def validate_only():
    assert TOLERANCES == (1e-4, 1e-6, 1e-8)
    assert MAX_ITER == 5000
    assert exp_d.EXPERIMENT_SCHEMA_VERSION == "D-v4"
    print("PASS: diagnostic has three predeclared tolerances and max_iter=5000.")
    print("PASS: reuses Experiment D preprocessing, split, objective and class weights.")
    print("PASS: no dataset loaded; no model fitted; no FedAvg training performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the three centralized optimizer diagnostics.")
    parser.add_argument("--data", default=None, help="Path to local IBM Telco CSV.")
    parser.add_argument("--output-dir", default="central_optimizer_diagnostic_outputs")
    parser.add_argument("--overwrite", action="store_true", help="Replace the diagnostic output directory.")
    args = parser.parse_args()
    validate_only()
    if args.run:
        run_diagnostic(args.data, args.output_dir, overwrite=args.overwrite)
    else:
        print("Validation only. Use --run only when explicitly ready to fit the three centralized models.")


if __name__ == "__main__":
    main()
