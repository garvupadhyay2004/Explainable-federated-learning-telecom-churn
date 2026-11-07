"""
Runnable starter experiment for the capstone:
Explainable Federated Learning Framework for Predicting Customer Churn in Telecom Networks.

This uses a centralized sklearn baseline and a simple NumPy FedAvg logistic-regression simulation.
"""
from __future__ import annotations
import argparse
import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv",
]

def load_data(path: str | None = None) -> pd.DataFrame:
    candidates = [path, "data/Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            print(f"Loading local dataset: {candidate}")
            return pd.read_csv(candidate)
    os.makedirs("data", exist_ok=True)
    for url in DATA_URLS:
        try:
            print(f"Downloading dataset from {url}")
            df = pd.read_csv(url)
            if "Churn" in df.columns:
                df.to_csv("data/Telco-Customer-Churn.csv", index=False)
                return df
        except Exception as exc:
            warnings.warn(f"Could not download from {url}: {exc}")
    raise RuntimeError(
        "Dataset download failed. Download the IBM Telco Customer Churn CSV "
        "and save it as data/Telco-Customer-Churn.csv, then rerun."
    )

def clean_data(df: pd.DataFrame):
    df = df.copy()
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' was not found in the CSV.")
    df = df.drop_duplicates()
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df.pop("Churn").astype(int)
    X = df
    return X, y

def make_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

def metric_dict(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["roc_auc"] = float("nan")
    return out, pred

def sigmoid(z):
    """Numerically stable sigmoid used by the federated implementation."""
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def weighted_logistic_objective(
    params,
    X,
    y,
    sample_weights,
    l2_strength=0.001,
):
    """Weighted mean logistic loss + (lambda / 2) * ||w||^2.

    The final parameter is the intercept and is deliberately not regularized.
    Returns the scalar objective and its analytical gradient.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)

    w = params[:-1]
    b = params[-1]
    total_weight = float(np.sum(sample_weights))
    if total_weight <= 0:
        raise ValueError("The sum of sample weights must be positive.")

    z = X @ w + b
    losses = np.logaddexp(0.0, z) - y * z
    data_loss = np.sum(sample_weights * losses) / total_weight
    regularization = 0.5 * l2_strength * np.dot(w, w)
    objective = data_loss + regularization

    error = sample_weights * (sigmoid(z) - y)
    grad_w = (X.T @ error) / total_weight + l2_strength * w
    grad_b = np.sum(error) / total_weight
    gradient = np.concatenate([grad_w, np.array([grad_b])])
    return float(objective), gradient


def train_fedavg(
    X_train,
    y_train,
    n_clients=3,
    rounds=12,
    local_epochs=3,
    learning_rate=0.08,
    seed=42,
    l2_strength=0.001,
):
    """FedAvg for binary logistic regression under a shared weighted objective.

    Class weights are computed globally from the training labels. Each client's
    local weighted loss is normalized by that client's sum of sample weights;
    client models are aggregated using those same sums as aggregation weights.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X_train must be 2D and have the same rows as y_train.")
    if n_clients < 1 or rounds < 1 or local_epochs < 1:
        raise ValueError("n_clients, rounds, and local_epochs must be positive.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("FedAvg expects binary labels encoded as 0 and 1.")
    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both binary classes.")

    # Global balanced weights, computed only from training labels.
    sample_weights = compute_sample_weight(
        class_weight="balanced", y=y
    ).astype(np.float64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    clients = [part for part in np.array_split(indices, n_clients) if len(part)]

    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    total_global_weight = float(np.sum(sample_weights))

    class_counts = {int(c): int(np.sum(y == c)) for c in (0, 1)}
    class_weight_values = {
        int(c): float(len(y) / (2.0 * class_counts[int(c)]))
        for c in (0, 1)
    }
    print(f"\nFederated simulation: {len(clients)} clients")
    print(f"Global class weights: {class_weight_values}")

    for rnd in range(1, rounds + 1):
        local_params = []
        client_weight_totals = []

        for idx in clients:
            Xc = X[idx]
            yc = y[idx]
            wc = sample_weights[idx]
            client_total_weight = float(np.sum(wc))

            local_w = weights.copy()
            local_b = float(bias)

            for _ in range(local_epochs):
                probabilities = sigmoid(Xc @ local_w + local_b)
                weighted_error = wc * (probabilities - yc)

                grad_w = (
                    Xc.T @ weighted_error
                ) / client_total_weight + l2_strength * local_w
                grad_b = float(np.sum(weighted_error) / client_total_weight)

                local_w -= learning_rate * grad_w
                local_b -= learning_rate * grad_b

            local_params.append((local_w, local_b))
            client_weight_totals.append(client_total_weight)

        # The aggregation coefficients sum to one and use the same client
        # weight totals that normalize the local weighted data objectives.
        weights = sum(
            local_w * client_weight
            for (local_w, _), client_weight
            in zip(local_params, client_weight_totals)
        ) / total_global_weight
        bias = sum(
            local_b * client_weight
            for (_, local_b), client_weight
            in zip(local_params, client_weight_totals)
        ) / total_global_weight

        if rnd in (1, 3, 6, rounds):
            global_objective, _ = weighted_logistic_objective(
                np.concatenate([weights, np.array([bias])]),
                X,
                y,
                sample_weights,
                l2_strength,
            )
            print(
                f"Round {rnd:02d}/{rounds} "
                f"| global weighted objective: {global_objective:.6f}"
            )

    return weights, bias

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to local churn CSV")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data)
    X, y = clean_data(df)
    print(f"\nDataset shape: {df.shape}")
    print("Target distribution (0=No churn, 1=Churn):")
    print(y.value_counts(normalize=True).sort_index().rename(index={0:"No",1:"Yes"}).round(3))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    preprocessor = make_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    # Convert sparse matrices to dense for the small demonstration dataset.
    X_train_t = X_train_t.toarray() if hasattr(X_train_t, "toarray") else np.asarray(X_train_t)
    X_test_t = X_test_t.toarray() if hasattr(X_test_t, "toarray") else np.asarray(X_test_t)

    print("\nTraining centralized Logistic Regression baseline...")
    L2_STRENGTH = 0.001
    central_sample_weights = compute_sample_weight(
        class_weight="balanced", y=y_train
    )
    total_sample_weight = float(np.sum(central_sample_weights))
    # For sklearn's lbfgs objective, this C maps the summed weighted loss
    # plus 1/(2C)||w||^2 to the weighted-mean loss plus
    # (L2_STRENGTH/2)||w||^2 used by the FedAvg implementation.
    central_C = 1.0 / (L2_STRENGTH * total_sample_weight)

    central = LogisticRegression(
        C=central_C,
        solver="lbfgs",
        max_iter=1500,
        class_weight="balanced",
        random_state=args.seed,
    )
    central.fit(X_train_t, y_train)
    print(f"Centralized L2 strength: {L2_STRENGTH:.6f}")
    print(f"Centralized LogisticRegression C: {central_C:.8f}")
    central_prob = central.predict_proba(X_test_t)[:, 1]
    central_metrics, central_pred = metric_dict(y_test, central_prob)
    print("\nCENTRALIZED RESULTS")
    for k, v in central_metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("\nCentralized classification report:")
    print(classification_report(y_test, central_pred, target_names=["No churn", "Churn"], zero_division=0))

    print("\nTraining federated logistic regression with FedAvg...")
    fw, fb = train_fedavg(
        X_train_t,
        y_train,
        n_clients=args.clients,
        rounds=args.rounds,
        seed=args.seed,
        l2_strength=L2_STRENGTH,
    )
    fed_prob = 1 / (1 + np.exp(-np.clip(X_test_t @ fw + fb, -35, 35)))
    fed_metrics, fed_pred = metric_dict(y_test, fed_prob)

    results = pd.DataFrame([
        {"model": "Centralized Logistic Regression", **central_metrics},
        {"model": "Federated Logistic Regression (FedAvg)", **fed_metrics},
    ])
    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/model_comparison.csv", index=False)
    print("\nMODEL COMPARISON (actual results from this run)")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, pred, title in zip(axes, [central_pred, fed_pred], ["Centralized", "Federated (FedAvg)"]):
        cm = confusion_matrix(y_test, pred)
        im = ax.imshow(cm)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["No churn", "Churn"])
        ax.set_yticks([0, 1], ["No churn", "Churn"])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrices.png", dpi=160)
    plt.close(fig)

    # Model explainability: permutation importance on the held-out set.
    print("\nCalculating permutation feature importance for centralized model...")
    perm = permutation_importance(
        central, X_test_t, y_test, scoring="f1", n_repeats=5,
        random_state=args.seed, n_jobs=-1
    )
    # Get transformed feature names from fitted preprocessing.
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(X_test_t.shape[1])])
    imp = pd.DataFrame({
        "feature": names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv("outputs/feature_importance.csv", index=False)
    top = imp.head(15).sort_values("importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.title("Permutation Feature Importance (centralized model, F1 scoring)")
    plt.xlabel("Decrease in F1 when feature is shuffled")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=160)
    plt.close()

    # SHAP explanations for the centralized Logistic Regression model.
    # LinearExplainer explains the model output in log-odds, not probability units.
    print("\nCalculating SHAP explanations for centralized model...")
    background_size = min(100, len(X_train_t))
    background = X_train_t[:background_size]
    shap_explainer = shap.LinearExplainer(central, background)
    shap_values = shap_explainer.shap_values(X_test_t)

    # Handle SHAP return formats across versions for binary classification.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    shap_global = pd.DataFrame({
        "feature": names,
        "mean_absolute_shap_value": np.mean(np.abs(shap_values), axis=0),
    }).sort_values("mean_absolute_shap_value", ascending=False)
    shap_global.to_csv("outputs/shap_global_importance.csv", index=False)

    top_shap = shap_global.head(15).sort_values("mean_absolute_shap_value")
    plt.figure(figsize=(9, 6))
    plt.barh(top_shap["feature"], top_shap["mean_absolute_shap_value"])
    plt.title("Global SHAP Importance (centralized model)")
    plt.xlabel("Mean absolute SHAP value (log-odds scale)")
    plt.tight_layout()
    plt.savefig("outputs/shap_global_importance.png", dpi=160)
    plt.close()

    # Explain the first held-out test example; positive SHAP values push
    # the model output toward churn, while negative values push it away.
    example_index = 0
    individual_shap = pd.DataFrame({
        "feature": names,
        "transformed_feature_value": X_test_t[example_index],
        "shap_value": shap_values[example_index],
    })
    individual_shap["absolute_shap_value"] = individual_shap["shap_value"].abs()
    individual_shap = individual_shap.sort_values("absolute_shap_value", ascending=False)
    individual_shap.to_csv("outputs/shap_individual_explanation.csv", index=False)

    top_individual = individual_shap.head(15).sort_values("shap_value")
    plt.figure(figsize=(9, 6))
    bar_colors = ["tab:blue" if value < 0 else "tab:orange" for value in top_individual["shap_value"]]
    plt.barh(top_individual["feature"], top_individual["shap_value"], color=bar_colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("SHAP Explanation for One Test Example (centralized model)")
    plt.xlabel("SHAP value (log-odds scale; positive pushes toward churn)")
    plt.tight_layout()
    plt.savefig("outputs/shap_individual_explanation.png", dpi=160)
    plt.close()
    print("SHAP global and individual explanation files generated.")

    # Save one illustrative individual prediction.
    example = {
        "actual_churn": int(np.asarray(y_test)[0]),
        "centralized_churn_probability": float(central_prob[0]),
        "centralized_prediction": int(central_pred[0]),
        "federated_churn_probability": float(fed_prob[0]),
        "federated_prediction": int(fed_pred[0]),
    }
    pd.DataFrame([example]).to_csv("outputs/example_prediction.csv", index=False)
    print("\nExample prediction:")
    print(pd.DataFrame([example]).to_string(index=False))
    print("\nSaved outputs:")
    print(" - outputs/model_comparison.csv")
    print(" - outputs/confusion_matrices.png")
    print(" - outputs/feature_importance.csv")
    print(" - outputs/feature_importance.png")
    print(" - outputs/shap_global_importance.csv")
    print(" - outputs/shap_global_importance.png")
    print(" - outputs/shap_individual_explanation.csv")
    print(" - outputs/shap_individual_explanation.png")
    print(" - outputs/example_prediction.csv")
    print("\nNote: the preprocessing transformer was fitted before the simulated client split.")
    print("This is a prototype limitation; report it transparently in your capstone.")

if __name__ == "__main__":
    main()
