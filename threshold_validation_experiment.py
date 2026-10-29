
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from train_with_shap import (
    load_data,
    clean_data,
    make_preprocessor,
    train_fedavg
)


SEED = 42
OUTPUT_DIR = "threshold_experiment_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities)
    }, predictions


# 1. Load and clean the dataset
df = load_data()
X, y = clean_data(df)

# 2. Keep a final test set untouched
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# 3. Split the remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.20,
    random_state=SEED,
    stratify=y_trainval
)

print("\nDataset split:")
print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_val)}")
print(f"Final test records: {len(X_test)}")

# 4. Fit preprocessing on training data only
preprocessor = make_preprocessor(X_train)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

X_train_t = to_dense(X_train_t)
X_val_t = to_dense(X_val_t)
X_test_t = to_dense(X_test_t)

# 5. Train centralized Logistic Regression
central = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    random_state=SEED
)
central.fit(X_train_t, y_train)

central_val_prob = central.predict_proba(X_val_t)[:, 1]
central_test_prob = central.predict_proba(X_test_t)[:, 1]

# 6. Train simulated FedAvg on training data only
fw, fb = train_fedavg(
    X_train_t,
    y_train,
    n_clients=3,
    rounds=12,
    local_epochs=3,
    learning_rate=0.08,
    seed=SEED
)

def fed_probabilities(X_input):
    scores = np.clip(X_input @ fw + fb, -35, 35)
    return 1 / (1 + np.exp(-scores))

fed_val_prob = fed_probabilities(X_val_t)
fed_test_prob = fed_probabilities(X_test_t)

# 7. Select each model's threshold using validation F1 only
thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)

def choose_threshold(y_val, probabilities):
    rows = []

    for threshold in thresholds:
        metrics, _ = get_metrics(y_val, probabilities, threshold)
        rows.append(metrics)

    validation_results = pd.DataFrame(rows)

    # If F1 ties, prefer the threshold closest to 0.5
    best_f1 = validation_results["f1"].max()
    tied = validation_results[
        np.isclose(validation_results["f1"], best_f1)
    ].copy()

    best_row = (
        tied.assign(distance=(tied["threshold"] - 0.5).abs())
        .sort_values("distance")
        .iloc[0]
    )

    return float(best_row["threshold"]), validation_results


central_threshold, central_val_results = choose_threshold(
    y_val, central_val_prob
)
fed_threshold, fed_val_results = choose_threshold(
    y_val, fed_val_prob
)

central_val_results.to_csv(
    f"{OUTPUT_DIR}/central_validation_thresholds.csv",
    index=False
)
fed_val_results.to_csv(
    f"{OUTPUT_DIR}/fedavg_validation_thresholds.csv",
    index=False
)

print("\nThresholds selected using validation data only:")
print(f"Centralized: {central_threshold:.2f}")
print(f"FedAvg:      {fed_threshold:.2f}")

# 8. Evaluate on the untouched test set
results = []
predictions_for_cm = {}

for model_name, probabilities, threshold in [
    ("Centralized - threshold 0.50", central_test_prob, 0.50),
    ("Centralized - tuned threshold", central_test_prob, central_threshold),
    ("FedAvg - threshold 0.50", fed_test_prob, 0.50),
    ("FedAvg - tuned threshold", fed_test_prob, fed_threshold)
]:
    metrics, predictions = get_metrics(
        y_test, probabilities, threshold
    )
    metrics["model"] = model_name
    results.append(metrics)
    predictions_for_cm[model_name] = predictions

results_df = pd.DataFrame(results)[
    ["model", "threshold", "accuracy", "precision",
     "recall", "f1", "roc_auc"]
]

results_df.to_csv(
    f"{OUTPUT_DIR}/test_threshold_comparison.csv",
    index=False
)

# Save confusion matrix counts for the final test set
cm_rows = []

for model_name, predictions in predictions_for_cm.items():
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    cm_rows.append({
        "model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

pd.DataFrame(cm_rows).to_csv(
    f"{OUTPUT_DIR}/test_confusion_matrices.csv",
    index=False
)

print("\nFINAL TEST RESULTS")
print(results_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

print(f"\nSaved new experiment files to: {OUTPUT_DIR}")
