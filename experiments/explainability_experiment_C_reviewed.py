"""Explainability Experiment C: matched centralized vs equal-size FedAvg.

CODE-REVIEW VERSION. This script does not run on import. Model fitting and
SHAP calculations occur only when invoked with the explicit --run flag.

Design:
- Reconstruct the exact Experiment B 64/16/20 split and preprocessing.
- Reproduce one centralized Logistic Regression and 20 equal-size FedAvg models.
- Validate every reproduced test probability against the preserved Experiment B
  prediction export before calculating explanations.
- Compute exact interventional SHAP values for linear models:
      phi_ij = (x_ij - mean(background_j)) * coefficient_j
  with base value = intercept + mean(background) @ coefficients.
  This is the exact linear-model SHAP solution under the independent/interventional
  feature-perturbation assumption; no approximate KernelExplainer is used.
- Produce transformed-feature and original-feature-grouped global explanations,
  deterministic individual explanations, ranking agreement, top-k overlap and
  across-seed stability summaries.

No existing project file is modified. Output is written to a new directory.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Reuse the exact validated partitioner and FedAvg implementation from Experiment B.
# Keep this companion file beside noniid_equal_size_experiment_export_validate.py.
try:
    import noniid_equal_size_experiment_export_validate as exp_b
except ImportError as exc:  # pragma: no cover - message is for user's local setup
    raise ImportError(
        "Place explainability_experiment_C.py beside "
        "noniid_equal_size_experiment_export_validate.py."
    ) from exc

ALPHAS = (10.0, 1.0, 0.5, 0.1)
SEEDS = (42, 43, 44, 45, 46)
N_CLIENTS = 3
L2_STRENGTH = 0.001
ROUNDS = 12
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.08
BACKGROUND_SIZE = 100
BACKGROUND_SEED = 20261106
INDIVIDUAL_COUNT = 8
INDIVIDUAL_SEED = 20261106
PREDICTION_ATOL = 1e-8
PREDICTION_RTOL = 1e-7
ADDITIVITY_ATOL = 1e-8

CENTRAL_NAME = "Centralized Logistic Regression"
FED_NAME = "FedAvg Equal-size Dirichlet"


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _load_and_prepare(data_path: str | None):
    candidates = [data_path, "data/Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    df = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            print(f"Loading dataset: {candidate}")
            df = pd.read_csv(candidate)
            break
    if df is None:
        raise FileNotFoundError(
            "IBM Telco CSV not found. Use --data PATH or place it at "
            "data/Telco-Customer-Churn.csv. No implicit download is performed."
        )
    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a Churn column.")

    df = df.drop_duplicates().copy()
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df.pop("Churn").astype(int)
    X = df

    # Exact two-stage stratified split used in Experiment B.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42,
        stratify=y_trainval,
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
    X_train_t = _dense(preprocessor.fit_transform(X_train)).astype(np.float64)
    X_val_t = _dense(preprocessor.transform(X_val)).astype(np.float64)
    X_test_t = _dense(preprocessor.transform(X_test)).astype(np.float64)

    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    if X_train_t.shape[1] != len(feature_names):
        raise AssertionError("Transformed feature count does not match feature names.")

    return {
        "preprocessor": preprocessor,
        "X_train_t": X_train_t,
        "X_val_t": X_val_t,
        "X_test_t": X_test_t,
        "y_train": y_train.to_numpy(dtype=np.int64),
        "y_val": y_val.to_numpy(dtype=np.int64),
        "y_test": y_test.to_numpy(dtype=np.int64),
        "test_row_indices": X_test.index.to_numpy(dtype=np.int64),
        "feature_names": feature_names,
        "train_row_count": len(X_train),
        "validation_row_count": len(X_val),
        "test_row_count": len(X_test),
        "original_columns": list(X.columns),
    }


def _read_reference_predictions(path: str) -> pd.DataFrame:
    ref = pd.read_csv(path)
    required = {
        "test_position", "test_row_index", "y_true", "model", "alpha", "seed",
        "predicted_probability", "validation_selected_threshold",
    }
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"Reference prediction CSV missing columns: {sorted(missing)}")
    if ref.duplicated(["model", "alpha", "seed", "test_position"]).any():
        raise ValueError("Reference prediction export contains duplicate model/test rows.")
    return ref


def _validate_reference_identity(data: dict, ref: pd.DataFrame) -> None:
    """Validate row identity and labels for every expected model group."""
    expected_positions = np.arange(data["test_row_count"], dtype=np.int64)
    expected_indices = data["test_row_indices"].astype(np.int64)
    expected_labels = data["y_test"].astype(np.int64)

    expected_groups = [(CENTRAL_NAME, None, 42)] + [
        (FED_NAME, alpha, seed) for alpha in ALPHAS for seed in SEEDS
    ]
    for model, alpha, seed in expected_groups:
        mask = ref["model"].eq(model) & ref["seed"].astype(int).eq(int(seed))
        if alpha is None:
            mask &= ref["alpha"].isna()
        else:
            mask &= np.isclose(pd.to_numeric(ref["alpha"], errors="coerce"), float(alpha))
        group = ref.loc[mask].sort_values("test_position")
        label = f"{model}, alpha={alpha}, seed={seed}"
        if len(group) != data["test_row_count"]:
            raise AssertionError(f"{label}: expected {data['test_row_count']} reference rows, found {len(group)}.")
        if group["test_position"].duplicated().any():
            raise AssertionError(f"{label}: duplicate test positions.")
        if not np.array_equal(group["test_position"].to_numpy(dtype=np.int64), expected_positions):
            raise AssertionError(f"{label}: test positions do not match canonical order.")
        if not np.array_equal(group["test_row_index"].to_numpy(dtype=np.int64), expected_indices):
            raise AssertionError(f"{label}: test row indices do not match canonical test set.")
        if not np.array_equal(group["y_true"].to_numpy(dtype=np.int64), expected_labels):
            raise AssertionError(f"{label}: actual labels do not match canonical test labels.")
        probabilities = pd.to_numeric(group["predicted_probability"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise AssertionError(f"{label}: invalid or non-finite reference probabilities.")


def _validate_probability_match(
    model: str, alpha: float | None, seed: int, probabilities: np.ndarray,
    ref: pd.DataFrame,
) -> float:
    mask = ref["model"].eq(model) & ref["seed"].astype(int).eq(int(seed))
    if alpha is None:
        mask &= ref["alpha"].isna()
    else:
        mask &= np.isclose(ref["alpha"].astype(float), float(alpha))
    group = ref.loc[mask].sort_values("test_position")
    if len(group) != len(probabilities):
        raise AssertionError(
            f"Reference rows mismatch for {model}, alpha={alpha}, seed={seed}: "
            f"expected {len(probabilities)}, found {len(group)}."
        )
    if not np.array_equal(group["test_position"].to_numpy(dtype=np.int64), np.arange(len(probabilities))):
        raise AssertionError(f"Test positions are not aligned for {model}, alpha={alpha}, seed={seed}.")
    expected = group["predicted_probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.isfinite(expected).all():
        raise AssertionError(f"Non-finite probabilities for {model}, alpha={alpha}, seed={seed}.")
    max_abs_diff = float(np.max(np.abs(probabilities - expected)))
    if not np.allclose(probabilities, expected, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL):
        raise AssertionError(
            f"Reproduced probabilities do not match Run 2 for {model}, alpha={alpha}, "
            f"seed={seed}; max_abs_diff={max_abs_diff:.3e}."
        )
    return max_abs_diff


def _analytical_interventional_shap(
    X_explain: np.ndarray, background: np.ndarray,
    coefficients: np.ndarray, intercept: float,
):
    """Exact interventional SHAP for a linear raw-logit model.

    With independent/interventional feature perturbation, the feature contribution
    is coefficient_j * (x_j - E_background[x_j]). This equals the exact SHAP
    solution for a linear model under this specified feature-independence game.
    """
    background_mean = np.mean(background, axis=0)
    values = (X_explain - background_mean) * coefficients.reshape(1, -1)
    base_value = float(intercept + background_mean @ coefficients)
    raw_output = X_explain @ coefficients + float(intercept)
    reconstructed = base_value + values.sum(axis=1)
    max_error = float(np.max(np.abs(raw_output - reconstructed)))
    if max_error > ADDITIVITY_ATOL:
        raise AssertionError(f"SHAP additivity check failed: max error={max_error:.3e}")
    return values, base_value, raw_output, max_error


def _group_mapping(feature_names: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    """Map transformed columns back to original input columns."""
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        if name.startswith("num__"):
            original = name[len("num__"):]
        elif name.startswith("cat__"):
            encoded = name[len("cat__"):]
            # Match the longest original column prefix; category text can contain underscores.
            candidates = [c for c in ORIGINAL_COLUMNS_CACHE if encoded.startswith(c + "_")]
            if candidates:
                original = max(candidates, key=len)
            else:
                original = encoded
        else:
            original = name
        groups.setdefault(original, []).append(idx)
    indices = [idx for values in groups.values() for idx in values]
    if sorted(indices) != list(range(len(feature_names))):
        raise AssertionError("Every transformed feature must map to exactly one original feature group.")
    if len(indices) != len(set(indices)):
        raise AssertionError("A transformed feature was assigned to more than one original feature group.")
    return list(groups), {key: np.asarray(value, dtype=int) for key, value in groups.items()}


ORIGINAL_COLUMNS_CACHE: list[str] = []


def _select_individual_positions(y_test: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic approximately class-stratified sample, independent of predictions."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    classes = np.unique(y_test)
    per_class = [count // len(classes)] * len(classes)
    for i in range(count % len(classes)):
        per_class[i] += 1
    for cls, n_take in zip(classes, per_class):
        candidates = np.flatnonzero(y_test == cls)
        chosen.extend(rng.choice(candidates, size=min(n_take, len(candidates)), replace=False).tolist())
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(len(y_test)), np.asarray(chosen, dtype=int))
        chosen.extend(rng.choice(remaining, size=count - len(chosen), replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=int))


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _topk_overlap(a_names: np.ndarray, a_values: np.ndarray,
                  b_names: np.ndarray, b_values: np.ndarray, k: int = 10) -> float:
    top_a = set(a_names[np.argsort(a_values)[-k:]])
    top_b = set(b_names[np.argsort(b_values)[-k:]])
    return len(top_a & top_b) / float(k)


def _plot_global(global_df: pd.DataFrame, out: Path) -> None:
    # Mean FedAvg importance over five seeds, shown separately for each alpha.
    fed = global_df[global_df["model"] == FED_NAME]
    central = global_df[global_df["model"] == CENTRAL_NAME]
    for alpha in ALPHAS:
        avg = fed[np.isclose(fed["alpha"].astype(float), alpha)].groupby("feature")[
            "mean_absolute_shap_value"
        ].mean()
        c = central.set_index("feature")["mean_absolute_shap_value"]
        plot = pd.DataFrame({"Centralized": c, "FedAvg mean": avg}).fillna(0.0)
        top = plot.max(axis=1).nlargest(15).sort_values()
        ax = plot.loc[top.index].plot(kind="barh", figsize=(10, 7))
        ax.set_title(f"Global SHAP comparison (alpha={alpha:g})")
        ax.set_xlabel("Mean absolute SHAP value (log-odds scale)")
        ax.set_ylabel("Transformed feature")
        plt.tight_layout()
        plt.savefig(out / f"global_shap_alpha_{str(alpha).replace('.', 'p')}.png", dpi=160)
        plt.close()


def _plot_individual_comparisons(
    individual_df: pd.DataFrame, individual_positions: np.ndarray, out: Path,
) -> None:
    """Plot centralized vs seed-mean FedAvg separately for each alpha.

    Never pool across alpha values: doing so would confound client-heterogeneity
    conditions and make the plot label scientifically misleading.
    """
    for alpha in ALPHAS:
        for position in individual_positions:
            subset = individual_df[
                (individual_df["test_position"] == int(position))
                & (individual_df["alpha"].isna() | np.isclose(individual_df["alpha"].fillna(-999).astype(float), alpha))
            ]
            central = subset[subset["model"] == CENTRAL_NAME].set_index("feature")["shap_value"]
            fed = subset[(subset["model"] == FED_NAME) & np.isclose(subset["alpha"].astype(float), alpha)]
            if len(fed["seed"].unique()) != len(SEEDS):
                raise AssertionError(f"Missing FedAvg seed explanations for alpha={alpha}, test_position={position}.")
            fed_mean = fed.groupby("feature")["shap_value"].mean()
            comparison = pd.DataFrame({"Centralized": central, f"FedAvg mean (5 seeds), alpha={alpha:g}": fed_mean}).fillna(0.0)
            top_features = comparison.abs().max(axis=1).nlargest(12).index
            comparison = comparison.loc[top_features]
            comparison = comparison.reindex(comparison.abs().max(axis=1).sort_values().index)
            ax = comparison.plot(kind="barh", figsize=(11, 7))
            ax.axvline(0.0, color="black", linewidth=0.8)
            label = int(subset["y_true"].iloc[0])
            ax.set_title(
                f"Individual SHAP | alpha={alpha:g} | test_position={int(position)} | actual_churn={label}"
            )
            ax.set_xlabel("SHAP value (log-odds; positive pushes toward churn)")
            ax.set_ylabel("Transformed feature")
            plt.tight_layout()
            alpha_label = str(alpha).replace(".", "p")
            plt.savefig(out / f"individual_shap_alpha_{alpha_label}_test_position_{int(position)}.png", dpi=160)
            plt.close()


def _validate_against_shap_library(
    X_explain: np.ndarray, background: np.ndarray,
    coefficients: np.ndarray, intercept: float,
    analytical_values: np.ndarray, analytical_base: float,
) -> float:
    """Cross-check a small sample against SHAP's interventional LinearExplainer."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "The SHAP package is required for the independent analytical cross-check. "
            "Install the project's SHAP dependency before running Experiment C."
        ) from exc

    sample_count = min(8, len(X_explain))
    sample = X_explain[:sample_count]
    explainer = shap.LinearExplainer(
        (coefficients, float(intercept)), background,
        feature_perturbation="interventional",
    )
    library_values = explainer.shap_values(sample)
    if isinstance(library_values, list):
        library_values = library_values[-1]
    library_values = np.asarray(library_values, dtype=float)
    if library_values.ndim == 3:
        library_values = library_values[:, :, -1]
    if library_values.shape != analytical_values[:sample_count].shape:
        raise AssertionError(
            f"SHAP library output shape {library_values.shape} does not match "
            f"analytical shape {analytical_values[:sample_count].shape}."
        )
    max_value_diff = float(np.max(np.abs(library_values - analytical_values[:sample_count])))
    if not np.allclose(library_values, analytical_values[:sample_count], rtol=1e-5, atol=1e-6):
        raise AssertionError(
            "Analytical SHAP values do not match SHAP LinearExplainer under the "
            f"interventional assumption; max difference={max_value_diff:.3e}."
        )
    expected_base = float(analytical_base)
    library_base = np.asarray(explainer.expected_value).reshape(-1)
    if library_base.size != 1 or not np.isclose(library_base[0], expected_base, rtol=1e-5, atol=1e-6):
        raise AssertionError("Analytical SHAP base value does not match SHAP LinearExplainer.")
    return max_value_diff


def _assert_finite_frames(*frames: pd.DataFrame) -> None:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number]).copy()
        # alpha is intentionally NaN for the single centralized reference model.
        if "alpha" in numeric.columns:
            numeric = numeric.drop(columns=["alpha"])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise AssertionError("A result table contains NaN or infinite numeric values outside the intentional centralized alpha field.")


def _write_metadata(out: Path, data: dict, background_positions: np.ndarray,
                    individual_positions: np.ndarray, max_prediction_diff: float,
                    max_additivity_error: float) -> None:
    metadata = {
        "experiment": "Explainability Experiment C",
        "status": "executed; results generated by this run",
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "n_clients": N_CLIENTS,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2_strength": L2_STRENGTH,
        "split": {"train": data["train_row_count"],
                  "validation": data["validation_row_count"],
                  "test": data["test_row_count"]},
        "transformed_feature_count": int(len(data["feature_names"])),
        "background_size": int(len(background_positions)),
        "background_seed": BACKGROUND_SEED,
        "background_train_positions": background_positions.tolist(),
        "individual_test_positions": individual_positions.tolist(),
        "individual_selection_seed": INDIVIDUAL_SEED,
        "prediction_match_max_absolute_difference": max_prediction_diff,
        "shap_additivity_max_absolute_error": max_additivity_error,
        "shap_method": "Exact analytical interventional SHAP for linear raw-logit models",
        "shap_output_space": "raw log-odds",
        "important_assumption": (
            "Interventional feature-independence assumption; transformed columns are perturbed independently. "
            "Grouped feature contributions sum dummy-variable SHAP values per original input feature."
        ),
        "limitations": [
            "Preprocessing is fitted centrally on the training split, as in Experiment B.",
            "This is a simulated FL experiment, not a production privacy guarantee.",
            "Eight individual records are illustrative; global metrics use all test records.",
            "Centralized model is one fitted reference, not five independent centralized seeds.",
        ],
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_experiment(data_path: str, reference_predictions: str, output_dir: str) -> None:
    out = Path(output_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(
            f"Output path already exists: {out}. Choose a new output directory; "
            "this script will not overwrite or merge existing results."
        )
    # Stage every artifact in a temporary sibling directory. Publish atomically
    # only after model, SHAP, grouping, and finite-value validations all pass.
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=str(out.parent)))
    work_out = staging
    data = _load_and_prepare(data_path)
    ref = _read_reference_predictions(reference_predictions)
    _validate_reference_identity(data, ref)

    global ORIGINAL_COLUMNS_CACHE
    ORIGINAL_COLUMNS_CACHE = list(data["original_columns"])
    group_names, group_indices = _group_mapping(data["feature_names"])

    X_train = data["X_train_t"]
    X_val = data["X_val_t"]
    X_test = data["X_test_t"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # Fixed training-only background sample, identical across all models.
    bg_rng = np.random.default_rng(BACKGROUND_SEED)
    background_positions = np.sort(bg_rng.choice(
        len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False
    ))
    background = X_train[background_positions]
    individual_positions = _select_individual_positions(
        y_test, INDIVIDUAL_COUNT, INDIVIDUAL_SEED
    )

    sample_weights = compute_sample_weight("balanced", y_train).astype(np.float64)
    total_weight = float(sample_weights.sum())
    central_c = 1.0 / (L2_STRENGTH * total_weight)
    central = LogisticRegression(
        C=central_c, penalty="l2", solver="lbfgs", max_iter=1500,
        class_weight="balanced", random_state=42,
    )
    central.fit(X_train, y_train)

    model_specs = [(CENTRAL_NAME, np.nan, 42, central.coef_.reshape(-1), float(central.intercept_[0]))]
    max_prediction_diff = 0.0
    max_additivity_error = 0.0
    max_shap_library_difference = 0.0

    central_prob = central.predict_proba(X_test)[:, 1]
    max_prediction_diff = max(max_prediction_diff, _validate_probability_match(
        CENTRAL_NAME, None, 42, central_prob, ref
    ))

    # Reproduce every FedAvg model from the same deterministic Experiment B setup.
    for alpha in ALPHAS:
        for seed in SEEDS:
            clients, _ = exp_b.constrained_dirichlet_label_partition(
                y_train, N_CLIENTS, alpha, seed
            )
            weights = compute_sample_weight("balanced", y_train).astype(np.float64)
            fw, fb, _history = exp_b.train_fedavg_partitioned(
                X_train, y_train, clients, weights,
                rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE, l2_strength=L2_STRENGTH,
            )
            probability = exp_b._sigmoid(X_test @ fw + fb)
            max_prediction_diff = max(max_prediction_diff, _validate_probability_match(
                FED_NAME, alpha, seed, probability, ref
            ))
            model_specs.append((FED_NAME, alpha, seed, fw, float(fb)))
            print(f"Validated reproduced FedAvg model: alpha={alpha:g}, seed={seed}")

    if len(model_specs) != 1 + len(ALPHAS) * len(SEEDS):
        raise AssertionError("Unexpected number of reproduced model configurations.")
    print(
        "PASS: all 21 reproduced model probability vectors match the preserved "
        f"Experiment B export (maximum absolute difference={max_prediction_diff:.3e})."
    )

    global_rows, grouped_global_rows = [], []
    individual_rows, individual_prediction_rows = [], []
    coefficient_rows = []
    per_model_importance: dict[tuple[str, float | None, int], np.ndarray] = {}
    per_model_group_importance: dict[tuple[str, float | None, int], np.ndarray] = {}

    for model, alpha, seed, coef, intercept in model_specs:
        shap_values, base_value, raw_output, additivity_error = _analytical_interventional_shap(
            X_test, background, coef, intercept
        )
        max_additivity_error = max(max_additivity_error, additivity_error)
        library_diff = _validate_against_shap_library(
            X_test, background, coef, intercept, shap_values, base_value
        )
        max_shap_library_difference = max(max_shap_library_difference, library_diff)
        importance = np.mean(np.abs(shap_values), axis=0)
        key = (model, None if pd.isna(alpha) else float(alpha), int(seed))
        per_model_importance[key] = importance

        for j, feature in enumerate(feature_names):
            global_rows.append({
                "model": model, "alpha": alpha, "seed": seed,
                "feature": feature, "mean_absolute_shap_value": float(importance[j]),
                "mean_signed_shap_value": float(np.mean(shap_values[:, j])),
                "coefficient": float(coef[j]),
            })
            coefficient_rows.append({
                "model": model, "alpha": alpha, "seed": seed,
                "feature": feature, "coefficient": float(coef[j]),
            })

        grouped_values = np.column_stack([
            shap_values[:, group_indices[name]].sum(axis=1) for name in group_names
        ])
        if not np.allclose(grouped_values.sum(axis=1), shap_values.sum(axis=1), rtol=1e-10, atol=1e-10):
            raise AssertionError("Grouped SHAP contributions do not reconstruct transformed-feature contributions.")
        grouped_importance = np.mean(np.abs(grouped_values), axis=0)
        per_model_group_importance[key] = grouped_importance
        for j, feature in enumerate(group_names):
            grouped_global_rows.append({
                "model": model, "alpha": alpha, "seed": seed,
                "feature": feature,
                "mean_absolute_grouped_shap_value": float(grouped_importance[j]),
                "mean_signed_grouped_shap_value": float(np.mean(grouped_values[:, j])),
            })

        for pos in individual_positions:
            individual_prediction_rows.append({
                "test_position": int(pos),
                "test_row_index": int(data["test_row_indices"][pos]),
                "y_true": int(y_test[pos]),
                "model": model, "alpha": alpha, "seed": seed,
                "predicted_probability": float(1.0 / (1.0 + np.exp(-np.clip(raw_output[pos], -35, 35)))),
                "raw_log_odds": float(raw_output[pos]),
                "shap_base_value": float(base_value),
            })
            for j, feature in enumerate(feature_names):
                individual_rows.append({
                    "test_position": int(pos),
                    "test_row_index": int(data["test_row_indices"][pos]),
                    "y_true": int(y_test[pos]),
                    "model": model, "alpha": alpha, "seed": seed,
                    "feature": feature,
                    "transformed_feature_value": float(X_test[pos, j]),
                    "shap_value": float(shap_values[pos, j]),
                    "absolute_shap_value": float(abs(shap_values[pos, j])),
                    "shap_base_value": float(base_value),
                    "raw_log_odds": float(raw_output[pos]),
                })

    global_df = pd.DataFrame(global_rows)
    grouped_df = pd.DataFrame(grouped_global_rows)
    individual_df = pd.DataFrame(individual_rows)
    individual_pred_df = pd.DataFrame(individual_prediction_rows)
    coeff_df = pd.DataFrame(coefficient_rows)

    # Ranking agreement and top-10 overlap against the one centralized reference.
    rank_rows = []
    central_imp = per_model_importance[(CENTRAL_NAME, None, 42)]
    central_group_imp = per_model_group_importance[(CENTRAL_NAME, None, 42)]
    for alpha in ALPHAS:
        for seed in SEEDS:
            fed_imp = per_model_importance[(FED_NAME, alpha, seed)]
            fed_group_imp = per_model_group_importance[(FED_NAME, alpha, seed)]
            rank_rows.append({
                "alpha": alpha, "seed": seed,
                "transformed_feature_spearman": _spearman(central_imp, fed_imp),
                "transformed_top10_overlap": _topk_overlap(
                    feature_names, central_imp, feature_names, fed_imp, 10
                ),
                "grouped_feature_spearman": _spearman(central_group_imp, fed_group_imp),
                "grouped_top10_overlap": _topk_overlap(
                    np.asarray(group_names), central_group_imp,
                    np.asarray(group_names), fed_group_imp, min(10, len(group_names))
                ),
            })
    rank_df = pd.DataFrame(rank_rows)

    # Seed stability: pairwise Spearman among five FedAvg seeds within each alpha.
    stability_rows = []
    for alpha in ALPHAS:
        for level, importance_store, names in (
            ("transformed", per_model_importance, feature_names),
            ("grouped", per_model_group_importance, np.asarray(group_names)),
        ):
            for i, seed_a in enumerate(SEEDS):
                for seed_b in SEEDS[i + 1:]:
                    a = importance_store[(FED_NAME, alpha, seed_a)]
                    b = importance_store[(FED_NAME, alpha, seed_b)]
                    stability_rows.append({
                        "alpha": alpha, "feature_level": level,
                        "seed_a": seed_a, "seed_b": seed_b,
                        "spearman_rank_correlation": _spearman(a, b),
                        "top10_overlap": _topk_overlap(names, a, names, b, min(10, len(names))),
                    })
    stability_df = pd.DataFrame(stability_rows)

    # Save files only after every model has reproduced the reference predictions.
    global_df.to_csv(work_out / "shap_global_by_model.csv", index=False)
    grouped_df.to_csv(work_out / "shap_global_grouped_by_model.csv", index=False)
    individual_df.to_csv(work_out / "shap_individual_values.csv", index=False)
    individual_pred_df.to_csv(work_out / "shap_individual_predictions.csv", index=False)
    rank_df.to_csv(work_out / "shap_rank_correlations.csv", index=False)
    stability_df.to_csv(work_out / "shap_seed_stability.csv", index=False)
    coeff_df.to_csv(work_out / "model_coefficients.csv", index=False)

    # Alpha-level aggregate summaries are descriptive means across five seeds.
    rank_df.groupby("alpha").agg(
        transformed_spearman_mean=("transformed_feature_spearman", "mean"),
        transformed_spearman_std=("transformed_feature_spearman", "std"),
        transformed_top10_overlap_mean=("transformed_top10_overlap", "mean"),
        grouped_spearman_mean=("grouped_feature_spearman", "mean"),
        grouped_spearman_std=("grouped_feature_spearman", "std"),
        grouped_top10_overlap_mean=("grouped_top10_overlap", "mean"),
    ).reset_index().to_csv(work_out / "shap_rank_summary_by_alpha.csv", index=False)
    stability_df.groupby(["alpha", "feature_level"]).agg(
        pairwise_spearman_mean=("spearman_rank_correlation", "mean"),
        pairwise_spearman_std=("spearman_rank_correlation", "std"),
        pairwise_top10_overlap_mean=("top10_overlap", "mean"),
    ).reset_index().to_csv(work_out / "shap_seed_stability_summary.csv", index=False)

    _plot_global(global_df, work_out)
    _plot_individual_comparisons(individual_df, individual_positions, work_out)
    _write_metadata(
        work_out, data, background_positions, individual_positions,
        max_prediction_diff, max_additivity_error,
    )
    metadata_path = work_out / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["shap_library_crosscheck_max_absolute_difference"] = max_shap_library_difference
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Validate all numeric result artifacts before publishing the staged folder.
    _assert_finite_frames(global_df, grouped_df, individual_df, individual_pred_df,
                          rank_df, stability_df, coeff_df)
    if out.exists():
        raise FileExistsError(f"Output path appeared during execution: {out}; refusing to overwrite.")
    os.replace(work_out, out)
    try:
        staging.rmdir()
    except OSError:
        pass
    print("PASS: SHAP additivity for all 21 models; maximum error="
          f"{max_additivity_error:.3e}.")
    print("PASS: analytical SHAP matches SHAP LinearExplainer; maximum value difference="
          f"{max_shap_library_difference:.3e}.")
    print(f"Saved Explainability Experiment C outputs to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explainability Experiment C: matched centralized vs FedAvg SHAP."
    )
    parser.add_argument("--data", default="data/Telco-Customer-Churn.csv",
                        help="Local IBM Telco churn CSV path.")
    parser.add_argument("--reference-predictions", default="noniid_equal_size_outputs_run2/equal_size_test_predictions.csv",
                        help="Preserved Experiment B Run 2 prediction export.")
    parser.add_argument("--output-dir", default="explainability_experiment_C_outputs",
                        help="New output directory; existing project files are not overwritten.")
    parser.add_argument("--run", action="store_true",
                        help="Explicitly reproduce 20 FedAvg models and calculate explanations.")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        print("\nNo training or SHAP computation was run. Add --run only after explicit approval.")
        return
    run_experiment(args.data, args.reference_predictions, args.output_dir)


if __name__ == "__main__":
    main()
