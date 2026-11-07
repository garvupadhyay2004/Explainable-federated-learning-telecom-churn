"""Read-only plotting functions for the telecom churn capstone dashboard.

All functions consume already-loaded evidence and return Matplotlib figures.
They do not train models, write files, or mutate the supplied DataFrames.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Evidence = Dict[str, Dict[str, pd.DataFrame]]


def _require_dataset(evidence: Evidence, experiment: str, dataset: str) -> pd.DataFrame:
    try:
        frame = evidence[experiment][dataset]
    except KeyError as exc:
        raise KeyError(
            f"Missing evidence dataset: experiment={experiment!r}, dataset={dataset!r}"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"Evidence dataset {experiment}/{dataset} is empty or invalid.")
    return frame.copy()


def _filter_alpha(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if "alpha" not in frame.columns:
        raise ValueError("Cannot filter by alpha: dataset has no 'alpha' column.")
    numeric_alpha = pd.to_numeric(frame["alpha"], errors="coerce")
    return frame.loc[np.isclose(numeric_alpha, float(alpha), equal_nan=False)].copy()


def _finish_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def performance_comparison_figure(
    evidence: Evidence,
    alpha: float,
    threshold_type: str = "validation_tuned",
    metric: str = "f1",
):
    """Compare centralized and FedAvg test metrics for one alpha/threshold policy."""
    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}")

    frame = _require_dataset(evidence, "B", "equal_size_metrics_by_seed")
    frame = frame.loc[frame["threshold_type"].astype(str) == str(threshold_type)].copy()
    if frame.empty:
        raise ValueError(f"No B metrics found for threshold_type={threshold_type!r}")

    central = frame.loc[frame["model"] == "Centralized Logistic Regression", metric]
    fed = _filter_alpha(
        frame.loc[frame["model"] == "FedAvg Equal-size Dirichlet"], alpha
    )[metric]
    if central.empty or fed.empty:
        raise ValueError("Required centralized or FedAvg metric rows are missing.")

    labels = ["Centralized", f"FedAvg (α={alpha:g})"]
    means = [float(central.mean()), float(fed.mean())]
    errors = [0.0, float(fed.std(ddof=1)) if len(fed) > 1 else 0.0]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(2)
    ax.bar(positions, means, yerr=errors, capsize=5)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1)
    _finish_axis(
        ax,
        f"Test {metric.upper()} — {threshold_type.replace('_', ' ')}",
        "Model",
        metric.upper(),
    )
    ax.text(
        0.02, 0.97,
        "Bars show mean across available rows; error bar shows FedAvg seed SD.",
        transform=ax.transAxes, va="top", fontsize=8,
    )
    fig.tight_layout()
    return fig


def metric_by_alpha_figure(
    evidence: Evidence,
    threshold_type: str = "validation_tuned",
    metric: str = "f1",
):
    """Show FedAvg mean and seed standard deviation across alpha settings."""
    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}")

    frame = _require_dataset(evidence, "B", "equal_size_metrics_by_seed")
    frame = frame.loc[
        (frame["model"] == "FedAvg Equal-size Dirichlet")
        & (frame["threshold_type"].astype(str) == str(threshold_type))
    ].copy()
    if frame.empty:
        raise ValueError("No FedAvg rows match the selected threshold policy.")

    grouped = frame.groupby("alpha", dropna=True)[metric].agg(["mean", "std"])
    grouped = grouped.sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        grouped.index.astype(float),
        grouped["mean"],
        yerr=grouped["std"].fillna(0),
        marker="o",
        capsize=4,
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    _finish_axis(
        ax,
        f"FedAvg {metric.upper()} across Dirichlet α",
        "Dirichlet α (log scale)",
        metric.upper(),
    )
    fig.tight_layout()
    return fig


def client_heterogeneity_figure(evidence: Evidence, alpha: float):
    """Plot measured label heterogeneity across the five available seeds."""
    frame = _require_dataset(evidence, "B", "equal_size_partition_diagnostics")
    frame = _filter_alpha(frame, alpha).sort_values("seed")
    if frame.empty:
        raise ValueError(f"No partition diagnostics found for alpha={alpha}.")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(frame["seed"].astype(str), frame["weighted_label_heterogeneity_sd"])
    _finish_axis(
        ax,
        f"Observed client label heterogeneity (α={alpha:g})",
        "Seed",
        "Weighted label heterogeneity SD",
    )
    fig.tight_layout()
    return fig


def convergence_figure(evidence: Evidence, alpha: float, seed: int):
    """Plot the existing Experiment B global objective by communication round."""
    frame = _require_dataset(evidence, "B", "equal_size_convergence_history")
    frame = _filter_alpha(frame, alpha)
    frame = frame.loc[frame["seed"] == seed].sort_values("round")
    if frame.empty:
        raise ValueError(f"No convergence history for alpha={alpha}, seed={seed}.")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(frame["round"], frame["global_weighted_objective"], marker="o", markersize=3)
    _finish_axis(
        ax,
        f"FedAvg training objective (α={alpha:g}, seed={seed})",
        "Communication round",
        "Global weighted objective",
    )
    fig.tight_layout()
    return fig


def shap_global_importance_figure(
    evidence: Evidence,
    alpha: float,
    model: str = "FedAvg Equal-size Dirichlet",
    top_n: int = 12,
    grouped: bool = True,
):
    """Plot mean absolute SHAP importance averaged across available seeds."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    dataset = "shap_global_grouped_by_model" if grouped else "shap_global_by_model"
    frame = _require_dataset(evidence, "C", dataset)
    frame = frame.loc[frame["model"] == model].copy()
    if model == "FedAvg Equal-size Dirichlet":
        frame = _filter_alpha(frame, alpha)
    elif model == "Centralized Logistic Regression":
        frame = frame.loc[pd.to_numeric(frame["alpha"], errors="coerce").isna()]
    else:
        raise ValueError(f"Unsupported model: {model!r}")
    value_col = (
        "mean_absolute_grouped_shap_value" if grouped
        else "mean_absolute_shap_value"
    )
    if frame.empty:
        raise ValueError("No SHAP rows match the selected model and alpha.")
    summary = (
        frame.groupby("feature", as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
        .head(top_n)
        .sort_values(value_col, ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.32 * len(summary))))
    ax.barh(summary["feature"], summary[value_col])
    level = "grouped original-feature" if grouped else "transformed-feature"
    _finish_axis(
        ax,
        f"Global {level} SHAP importance — {model}"
        + (f", α={alpha:g}" if model == "FedAvg Equal-size Dirichlet" else ""),
        "Mean absolute SHAP value (raw log-odds scale)",
        "Feature",
    )
    fig.tight_layout()
    return fig


def shap_seed_stability_figure(evidence: Evidence, alpha: float):
    """Show pairwise explanation-rank stability across FedAvg seeds."""
    frame = _require_dataset(evidence, "C", "shap_seed_stability_summary")
    frame = _filter_alpha(frame, alpha)
    if frame.empty:
        raise ValueError(f"No SHAP stability summary for alpha={alpha}.")
    labels = frame["feature_level"].astype(str).tolist()
    values = frame["pairwise_spearman_mean"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, values)
    ax.set_ylim(-1, 1)
    _finish_axis(
        ax,
        f"SHAP feature-rank stability across seeds (α={alpha:g})",
        "Feature representation",
        "Mean pairwise Spearman correlation",
    )
    fig.tight_layout()
    return fig


def central_fed_explanation_agreement_figure(evidence: Evidence):
    """Plot central-vs-FedAvg SHAP rank agreement across alpha values."""
    frame = _require_dataset(evidence, "C", "shap_rank_summary_by_alpha")
    frame = frame.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(frame))
    ax.plot(
        x, frame["transformed_spearman_mean"],
        marker="o", label="Transformed features",
    )
    ax.plot(
        x, frame["grouped_spearman_mean"],
        marker="s", label="Grouped original features",
    )
    ax.set_xticks(x, [f"{v:g}" for v in frame["alpha"]])
    ax.set_ylim(-1, 1)
    ax.legend()
    _finish_axis(
        ax,
        "Centralized vs FedAvg explanation-rank agreement",
        "Dirichlet α",
        "Mean Spearman rank correlation",
    )
    fig.tight_layout()
    return fig


def optimization_config_figure(
    evidence: Evidence,
    alpha: float,
    metric: str = "final_validation_log_loss",
):
    """Compare Experiment D optimization configurations across seeds."""
    allowed = {
        "final_global_objective",
        "final_global_gradient_l2_norm",
        "objective_gap_to_central",
        "final_coefficient_l2_distance_to_central",
        "final_train_probability_rmse_vs_central",
        "final_validation_log_loss",
        "final_validation_roc_auc",
    }
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    frame = _require_dataset(evidence, "D", "experiment_D_final_summary")
    frame = _filter_alpha(frame, alpha)
    if frame.empty:
        raise ValueError(f"No Experiment D runs found for alpha={alpha}.")
    summary = frame.groupby("config")[metric].agg(["mean", "std"]).sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(summary))
    ax.bar(
        positions, summary["mean"],
        yerr=summary["std"].fillna(0), capsize=4,
    )
    ax.set_xticks(positions, summary.index, rotation=20, ha="right")
    _finish_axis(
        ax,
        f"Experiment D configuration comparison (α={alpha:g})",
        "Optimization configuration",
        metric.replace("_", " ").title(),
    )
    fig.tight_layout()
    return fig


def optimization_history_figure(
    evidence: Evidence,
    alpha: float,
    seed: int,
    config: str,
    metric: str = "global_objective",
):
    """Plot one existing Experiment D run over its recorded rounds."""
    allowed = {
        "global_objective",
        "global_gradient_l2_norm",
        "coefficient_l2_distance_to_central",
        "train_probability_rmse_vs_central",
        "validation_log_loss",
        "validation_roc_auc",
        "objective_gap_to_central",
    }
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    frame = _require_dataset(evidence, "D", "experiment_D_round_history")
    frame = _filter_alpha(frame, alpha)
    frame = frame.loc[
        (frame["seed"] == seed) & (frame["config"].astype(str) == str(config))
    ].sort_values("round")
    if frame.empty:
        raise ValueError(
            f"No Experiment D history for alpha={alpha}, seed={seed}, config={config}."
        )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(frame["round"], frame[metric])
    _finish_axis(
        ax,
        f"Experiment D — {metric.replace('_', ' ')} ({config}, α={alpha:g}, seed={seed})",
        "Communication round",
        metric.replace("_", " ").title(),
    )
    fig.tight_layout()
    return fig
