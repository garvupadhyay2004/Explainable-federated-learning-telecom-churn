"""Mathematical checks for the aligned centralized/FedAvg logistic objective.

This script uses only a small synthetic dataset. It does NOT load the churn
CSV, train the capstone models, or write/overwrite experiment outputs.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from train_with_shap import sigmoid, weighted_logistic_objective

L2 = 0.001
SEED = 42


def finite_difference_gradient(params, X, y, sample_weights, eps=1e-6):
    numerical = np.zeros_like(params, dtype=np.float64)
    for j in range(len(params)):
        step = np.zeros_like(params, dtype=np.float64)
        step[j] = eps
        f_plus, _ = weighted_logistic_objective(
            params + step, X, y, sample_weights, L2
        )
        f_minus, _ = weighted_logistic_objective(
            params - step, X, y, sample_weights, L2
        )
        numerical[j] = (f_plus - f_minus) / (2.0 * eps)
    return numerical


def main():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(120, 5))
    true_w = np.array([0.7, -0.4, 0.2, 0.0, 0.5])
    logits = X @ true_w - 0.15
    probabilities = sigmoid(logits)
    y = (rng.random(len(X)) < probabilities).astype(int)

    # Ensure both classes exist in this synthetic test.
    if len(np.unique(y)) != 2:
        raise RuntimeError("Synthetic validation data unexpectedly has one class.")

    sample_weights = compute_sample_weight("balanced", y).astype(np.float64)
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    expected_w0 = len(y) / (2.0 * n0)
    expected_w1 = len(y) / (2.0 * n1)

    print("CHECK 1 — Analytical gradient vs finite differences")
    params = rng.normal(scale=0.2, size=X.shape[1] + 1)
    _, analytical = weighted_logistic_objective(params, X, y, sample_weights, L2)
    numerical = finite_difference_gradient(params, X, y, sample_weights)
    max_abs_error = float(np.max(np.abs(analytical - numerical)))
    print(f"Maximum absolute gradient error: {max_abs_error:.3e}")
    assert max_abs_error < 1e-5, "Gradient check failed."
    print("PASS\n")

    print("CHECK 2 — Balanced class weights")
    actual_w0 = float(sample_weights[y == 0][0])
    actual_w1 = float(sample_weights[y == 1][0])
    print(f"Class 0 count={n0}, expected weight={expected_w0:.8f}, actual={actual_w0:.8f}")
    print(f"Class 1 count={n1}, expected weight={expected_w1:.8f}, actual={actual_w1:.8f}")
    assert np.isclose(actual_w0, expected_w0) and np.isclose(actual_w1, expected_w1)
    assert np.isclose(sample_weights.sum(), len(y))
    print(f"Total sample weight={sample_weights.sum():.8f}; N={len(y)}")
    print("PASS\n")

    print("CHECK 3 — Client aggregation reproduces the global gradient")
    client_ids = np.array_split(rng.permutation(len(y)), 3)
    global_params = rng.normal(scale=0.1, size=X.shape[1] + 1)
    _, global_gradient = weighted_logistic_objective(
        global_params, X, y, sample_weights, L2
    )
    aggregated_gradient = np.zeros_like(global_gradient)
    aggregation_coefficients = []
    total_weight = float(sample_weights.sum())

    for ids in client_ids:
        client_weight = float(sample_weights[ids].sum())
        aggregation_coefficients.append(client_weight / total_weight)
        _, client_gradient = weighted_logistic_objective(
            global_params, X[ids], y[ids], sample_weights[ids], L2
        )
        aggregated_gradient += (client_weight / total_weight) * client_gradient

    agg_error = float(np.max(np.abs(global_gradient - aggregated_gradient)))
    coefficient_sum = float(np.sum(aggregation_coefficients))
    print(f"Aggregation coefficients: {np.round(aggregation_coefficients, 8)}")
    print(f"Coefficient sum: {coefficient_sum:.12f}")
    print(f"Maximum global-vs-aggregated gradient error: {agg_error:.3e}")
    assert np.isclose(coefficient_sum, 1.0)
    assert agg_error < 1e-10, "Client aggregation does not reproduce global gradient."
    print("PASS\n")

    print("CHECK 4 — sklearn C mapping against the same objective")
    total_weight = float(sample_weights.sum())
    C_value = 1.0 / (L2 * total_weight)
    sklearn_model = LogisticRegression(
        C=C_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=10000,
        tol=1e-12,
        class_weight="balanced",
        fit_intercept=True,
    )
    sklearn_model.fit(X, y)

    # Independently minimize the declared weighted-mean objective.
    def scipy_objective(p):
        value, gradient = weighted_logistic_objective(p, X, y, sample_weights, L2)
        return value, gradient

    scipy_result = minimize(
        scipy_objective,
        np.zeros(X.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 20000, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
    )
    sklearn_params = np.concatenate([sklearn_model.coef_.ravel(), sklearn_model.intercept_])
    coefficient_error = float(np.max(np.abs(sklearn_params - scipy_result.x)))
    sklearn_objective, _ = weighted_logistic_objective(
        sklearn_params, X, y, sample_weights, L2
    )
    scipy_objective_value, _ = weighted_logistic_objective(
        scipy_result.x, X, y, sample_weights, L2
    )
    print(f"N={len(y)}, total sample weight W={total_weight:.8f}")
    print(f"Mapped C=1/(lambda*W): {C_value:.8f}")
    print(f"SciPy optimizer success: {scipy_result.success}; message: {scipy_result.message}")
    print(f"Maximum coefficient difference: {coefficient_error:.3e}")
    print(f"sklearn parameters objective: {sklearn_objective:.12f}")
    print(f"SciPy optimum objective:       {scipy_objective_value:.12f}")
    print(f"sklearn n_iter_: {sklearn_model.n_iter_}")
    assert scipy_result.success, "Independent SciPy optimizer did not converge."
    assert coefficient_error < 5e-4, (
        "sklearn and the declared objective do not agree closely enough; "
        "do not claim objective alignment until this is investigated."
    )
    print("PASS\n")

    print("ALL MATHEMATICAL VALIDATION CHECKS PASSED.")
    print("This validates the objective on synthetic data only; it does not train the churn models.")


if __name__ == "__main__":
    main()
