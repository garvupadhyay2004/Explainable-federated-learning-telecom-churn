"""Read-only Streamlit dashboard for the telecom churn capstone.

The dashboard loads existing B/C/D experiment evidence only. It does not
launch training scripts, download datasets, or write to experiment folders.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import get_evidence_inventory, load_all_evidence
import visualizations as viz


st.set_page_config(
    page_title="Explainable Federated Churn",
    page_icon="📊",
    layout="wide",
)

CENTRAL_MODEL = "Centralized Logistic Regression"
FED_MODEL = "FedAvg Equal-size Dirichlet"
METRICS = ["f1", "roc_auc", "precision", "recall", "accuracy"]
THRESHOLD_TYPES = ["fixed_0.50", "validation_tuned"]


@st.cache_data(show_spinner="Loading and validating existing experiment evidence...")
def load_evidence_cached():
    """Cache the validated CSV evidence in Streamlit's data cache."""
    return load_all_evidence()


def show_figure(figure) -> None:
    """Render a Matplotlib figure and release its resources."""
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def format_metric(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


st.title("Explainable Federated Learning for Telecom Churn")
st.caption(
    "Read-only research dashboard | Experiments B, C and D | "
    "Simulated federated clients — not a production privacy guarantee"
)

try:
    evidence = load_evidence_cached()
except Exception as exc:
    st.error("The experiment evidence could not be loaded or validated.")
    st.exception(exc)
    st.stop()

b_metrics = evidence["B"]["equal_size_metrics_by_seed"]
c_shap = evidence["C"]["shap_global_grouped_by_model"]
d_summary = evidence["D"]["experiment_D_final_summary"]

alpha_values = sorted(
    pd.to_numeric(
        b_metrics.loc[b_metrics["model"] == FED_MODEL, "alpha"],
        errors="coerce",
    ).dropna().unique().tolist()
)
seed_values = sorted(
    pd.to_numeric(b_metrics.loc[b_metrics["model"] == FED_MODEL, "seed"])
    .dropna().astype(int).unique().tolist()
)
d_seed_values = sorted(d_summary["seed"].dropna().astype(int).unique().tolist())
d_config_values = sorted(d_summary["config"].astype(str).unique().tolist())

with st.sidebar:
    st.header("Dashboard filters")
    selected_alpha = st.selectbox(
        "Dirichlet alpha (α)",
        alpha_values,
        index=alpha_values.index(0.1) if 0.1 in alpha_values else 0,
        format_func=lambda value: f"{value:g}",
    )
    selected_seed = st.selectbox(
        "Seed",
        seed_values,
        index=seed_values.index(42) if 42 in seed_values else 0,
    )
    selected_threshold = st.selectbox(
        "Evaluation threshold policy",
        THRESHOLD_TYPES,
        index=1,
        format_func=lambda value: (
            "Validation-tuned threshold"
            if value == "validation_tuned"
            else "Fixed threshold = 0.50"
        ),
    )
    selected_metric = st.selectbox(
        "Performance metric",
        METRICS,
        index=0,
        format_func=lambda value: value.upper(),
    )
    st.divider()
    st.caption("Experiment D controls")
    selected_d_seed = st.selectbox(
        "Optimization seed",
        d_seed_values,
        index=d_seed_values.index(42) if 42 in d_seed_values else 0,
    )
    selected_d_config = st.selectbox(
        "Optimization configuration",
        d_config_values,
        index=d_config_values.index("R12_E3")
        if "R12_E3" in d_config_values else 0,
    )

tabs = st.tabs(
    [
        "Project Overview",
        "Federated Performance & Heterogeneity",
        "Explainability",
        "Convergence Analysis",
        "Methodology & Limitations",
    ]
)

# ---------------------------------------------------------------------------
# 1. Project overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Project Overview")
    st.write(
        "This project evaluates a simulated FedAvg framework for telecom "
        "customer-churn prediction, studies client-label heterogeneity, and "
        "examines model explanations and optimization behavior."
    )

    inventory = get_evidence_inventory(evidence)
    total_rows = int(inventory["rows"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiments", "3")
    col2.metric("Validated datasets", f"{len(inventory)}")
    col3.metric("Evidence records", f"{total_rows:,}")
    col4.metric("Federated clients", "3")

    st.subheader("Experiment map")
    overview = pd.DataFrame(
        [
            {
                "Experiment": "B",
                "Focus": "Federated performance and non-IID client heterogeneity",
                "Datasets": len(evidence["B"]),
                "Records": int(inventory.loc[inventory["experiment"] == "B", "rows"].sum()),
            },
            {
                "Experiment": "C",
                "Focus": "Global and individual SHAP explanations and stability",
                "Datasets": len(evidence["C"]),
                "Records": int(inventory.loc[inventory["experiment"] == "C", "rows"].sum()),
            },
            {
                "Experiment": "D",
                "Focus": "Optimization behavior across rounds and local epochs",
                "Datasets": len(evidence["D"]),
                "Records": int(inventory.loc[inventory["experiment"] == "D", "rows"].sum()),
            },
        ]
    )
    st.dataframe(overview, hide_index=True, use_container_width=True)

    st.subheader("Evidence inventory")
    st.dataframe(
        inventory.sort_values(["experiment", "dataset"]),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "The dashboard reads precomputed CSV evidence. It does not train or "
        "retrain models, execute experiment scripts, download data, or write "
        "to the experiment output directories."
    )

# ---------------------------------------------------------------------------
# 2. Performance and heterogeneity
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Federated Performance & Heterogeneity")
    st.caption(
        f"Selected α={selected_alpha:g} | "
        f"Threshold policy: {selected_threshold.replace('_', ' ')}"
    )

    metric_frame = b_metrics.loc[
        b_metrics["threshold_type"].astype(str) == selected_threshold
    ].copy()
    central_rows = metric_frame.loc[metric_frame["model"] == CENTRAL_MODEL]
    fed_rows = metric_frame.loc[
        (metric_frame["model"] == FED_MODEL)
        & (pd.to_numeric(metric_frame["alpha"], errors="coerce") == selected_alpha)
    ]

    if not central_rows.empty and not fed_rows.empty:
        central_value = float(central_rows[selected_metric].mean())
        fed_value = float(fed_rows[selected_metric].mean())
        delta = fed_value - central_value

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Centralized {selected_metric.upper()}", format_metric(central_value))
        c2.metric(f"FedAvg {selected_metric.upper()}", format_metric(fed_value))
        c3.metric(
            "FedAvg − Centralized",
            format_metric(delta),
            delta=format_metric(delta),
            delta_color="normal",
        )
    else:
        st.warning("No matching rows are available for the selected filters.")

    left, right = st.columns(2)
    with left:
        st.subheader("Centralized vs. FedAvg")
        show_figure(
            viz.performance_comparison_figure(
                evidence,
                selected_alpha,
                threshold_type=selected_threshold,
                metric=selected_metric,
            )
        )
    with right:
        st.subheader("FedAvg performance across alpha")
        show_figure(
            viz.metric_by_alpha_figure(
                evidence,
                threshold_type=selected_threshold,
                metric=selected_metric,
            )
        )

    st.subheader("Client label heterogeneity")
    show_figure(viz.client_heterogeneity_figure(evidence, selected_alpha))

    st.subheader("Per-seed test metrics")
    display_columns = [
        "model", "alpha", "seed", "threshold_type",
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "validation_selected_threshold",
    ]
    selected_rows = metric_frame.loc[
        (metric_frame["model"] == CENTRAL_MODEL)
        | (
            (metric_frame["model"] == FED_MODEL)
            & (pd.to_numeric(metric_frame["alpha"], errors="coerce") == selected_alpha)
        )
    ]
    st.dataframe(
        selected_rows[display_columns].sort_values(["model", "seed"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Test confusion-matrix counts")
    confusion = evidence["B"]["equal_size_test_confusion_matrices"]
    confusion = confusion.loc[
        (confusion["threshold_type"].astype(str) == selected_threshold)
        & (
            (confusion["model"] == CENTRAL_MODEL)
            | (
                (confusion["model"] == FED_MODEL)
                & (pd.to_numeric(confusion["alpha"], errors="coerce") == selected_alpha)
            )
        )
    ]
    st.dataframe(confusion, hide_index=True, use_container_width=True)

    st.warning(
        "Interpretation: the observed results do not establish that FedAvg is "
        "superior to centralized training. Centralized ROC-AUC is higher in "
        "these experiments, and the paired F1 confidence intervals reported "
        "in the experiment analysis include zero."
    )

# ---------------------------------------------------------------------------
# 3. Explainability
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Explainability")
    st.caption(
        "SHAP values are calculated on the raw log-odds scale using an "
        "interventional feature-independence assumption."
    )

    explain_model = st.radio(
        "Model",
        [FED_MODEL, CENTRAL_MODEL],
        horizontal=True,
        format_func=lambda value: "FedAvg" if value == FED_MODEL else "Centralized",
    )
    grouped = st.toggle(
        "Group transformed columns by original feature",
        value=True,
        help=(
            "Grouped contributions sum dummy-variable SHAP values by original "
            "feature; this is not coalition-level grouped SHAP."
        ),
    )
    top_n = st.slider("Number of features", min_value=5, max_value=20, value=12)

    show_figure(
        viz.shap_global_importance_figure(
            evidence,
            selected_alpha,
            model=explain_model,
            top_n=top_n,
            grouped=grouped,
        )
    )

    left, right = st.columns(2)
    with left:
        st.subheader("FedAvg explanation stability across seeds")
        show_figure(viz.shap_seed_stability_figure(evidence, selected_alpha))
    with right:
        st.subheader("Centralized vs. FedAvg explanation agreement")
        show_figure(viz.central_fed_explanation_agreement_figure(evidence))

    st.subheader("Individual explanation examples")
    individual_predictions = evidence["C"]["shap_individual_predictions"]
    individual_values = evidence["C"]["shap_individual_values"]

    individual_options = (
        individual_predictions.loc[
            individual_predictions["model"] == FED_MODEL
        ]
        .loc[
            pd.to_numeric(
                individual_predictions.loc[
                    individual_predictions["model"] == FED_MODEL, "alpha"
                ],
                errors="coerce",
            ) == selected_alpha
        ][["test_position", "seed"]]
        .drop_duplicates()
        .sort_values(["test_position", "seed"])
    )
    if not individual_options.empty:
        example_labels = [
            f"Test position {int(row.test_position)} | seed {int(row.seed)}"
            for row in individual_options.itertuples(index=False)
        ]
        chosen_label = st.selectbox("Illustrative FedAvg example", example_labels)
        chosen_index = example_labels.index(chosen_label)
        chosen = individual_options.iloc[chosen_index]

        prediction = individual_predictions.loc[
            (individual_predictions["model"] == FED_MODEL)
            & (pd.to_numeric(individual_predictions["alpha"], errors="coerce") == selected_alpha)
            & (individual_predictions["test_position"] == chosen["test_position"])
            & (individual_predictions["seed"] == chosen["seed"])
        ]
        values = individual_values.loc[
            (individual_values["model"] == FED_MODEL)
            & (pd.to_numeric(individual_values["alpha"], errors="coerce") == selected_alpha)
            & (individual_values["test_position"] == chosen["test_position"])
            & (individual_values["seed"] == chosen["seed"])
        ].copy()

        if not prediction.empty:
            st.dataframe(prediction, hide_index=True, use_container_width=True)
        if not values.empty:
            values = values.sort_values("absolute_shap_value", ascending=False)
            st.dataframe(
                values[
                    [
                        "feature", "transformed_feature_value",
                        "shap_value", "absolute_shap_value",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("No individual FedAvg explanation examples match this alpha.")

    st.info(
        "The eight individual test examples are illustrative, not a "
        "population-level explanation sample. SHAP rank stability across "
        "FedAvg seeds does not imply agreement with the centralized model."
    )

# ---------------------------------------------------------------------------
# 4. Convergence and optimization
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Convergence Analysis")
    st.subheader("Experiment B — FedAvg training objective")
    show_figure(viz.convergence_figure(evidence, selected_alpha, selected_seed))

    st.subheader("Experiment D — Configuration comparison")
    d_metric = st.selectbox(
        "Experiment D comparison metric",
        [
            "final_validation_log_loss",
            "final_validation_roc_auc",
            "final_global_objective",
            "final_global_gradient_l2_norm",
            "objective_gap_to_central",
            "final_coefficient_l2_distance_to_central",
            "final_train_probability_rmse_vs_central",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    show_figure(
        viz.optimization_config_figure(
            evidence,
            selected_alpha,
            metric=d_metric,
        )
    )

    st.subheader("Experiment D — Selected run history")
    d_history_metric = st.selectbox(
        "Round-history metric",
        [
            "global_objective",
            "global_gradient_l2_norm",
            "objective_gap_to_central",
            "coefficient_l2_distance_to_central",
            "train_probability_rmse_vs_central",
            "validation_log_loss",
            "validation_roc_auc",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    show_figure(
        viz.optimization_history_figure(
            evidence,
            selected_alpha,
            selected_d_seed,
            selected_d_config,
            metric=d_history_metric,
        )
    )

    d_selected = d_summary.loc[
        (pd.to_numeric(d_summary["alpha"], errors="coerce") == selected_alpha)
        & (d_summary["seed"] == selected_d_seed)
        & (d_summary["config"].astype(str) == selected_d_config)
    ]
    if not d_selected.empty:
        st.subheader("Selected run summary")
        st.dataframe(d_selected, hide_index=True, use_container_width=True)

    st.warning(
        "Experiment D uses validation metrics and optimization diagnostics; "
        "the test set was not used in this experiment. A lower objective or "
        "gradient norm indicates optimization progress, not by itself better "
        "generalization."
    )

# ---------------------------------------------------------------------------
# 5. Methodology and limitations
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Methodology & Limitations")

    st.subheader("Dataset and split")
    st.markdown(
        """
        - Dataset: IBM Telco Customer Churn, 7,043 records and 21 columns.
        - Target prevalence: approximately 26.5% churn and 73.5% non-churn.
        - Stratified split: 4,507 training, 1,127 validation and 1,409 test records.
        - Federated simulation: three clients with equal-sized partitions and
          Dirichlet-controlled label heterogeneity.
        """
    )

    st.subheader("Experiment B — performance")
    st.markdown(
        """
        - Compares a centralized Logistic Regression reference with simulated
          FedAvg under multiple Dirichlet alpha values and seeds.
        - Reports fixed-threshold (0.50) and validation-tuned evaluation.
        - Threshold selection uses validation data; test metrics are reported
          after threshold selection.
        - Five seeds provide limited evidence about run-to-run variability.
        """
    )

    st.subheader("Experiment C — explainability")
    st.markdown(
        """
        - Computes analytical interventional SHAP for linear raw-logit models.
        - Uses a training-background sample of 100 rows.
        - Global mean absolute SHAP importance is descriptive and calculated
          on the test set.
        - Grouped feature contributions sum transformed-column SHAP values
          by original feature; they are not coalition-level grouped SHAP.
        - Individual explanations cover eight illustrative test examples.
        """
    )

    st.subheader("Experiment D — optimization")
    st.markdown(
        """
        - Compares communication rounds and local full-batch gradient steps.
        - Uses five configurations: R12_E3, R50_E3, R100_E3, R50_E1 and R50_E5.
        - Tracks global objective, gradient norm, coefficient distance,
          probability differences and validation metrics.
        - No test-set metrics are used in Experiment D.
        """
    )

    st.subheader("Limitations that matter")
    st.markdown(
        """
        1. **Simulation is not deployment.** Clients are simulated in one
           experimental environment; no secure aggregation, differential
           privacy, or production network isolation is demonstrated.
        2. **Preprocessing is centralized.** The preprocessing pipeline and
           class-weight calculation are not a fully decentralized protocol.
        3. **Optimization is incomplete in some configurations.** The
           12-round, 3-local-epoch models analyzed in Experiment C are not
           equivalent to the longer optimization runs in Experiment D.
        4. **Explanation assumptions matter.** Interventional SHAP assumes
           feature independence during perturbation, which can be unrealistic
           for correlated telecom features.
        5. **Small seed count.** Five seeds are insufficient for broad
           statistical generalization.
        6. **No claim of federated superiority.** Results show trade-offs;
           they do not establish that FedAvg consistently outperforms the
           centralized reference.
        """
    )

    st.subheader("Reproducibility and evidence")
    st.write(
        "All displayed results are read from the validated experiment CSVs. "
        "The dashboard does not regenerate evidence. Refer to the experiment "
        "scripts and retained reports for the full implementation details."
    )

st.divider()
st.caption(
    "Capstone research prototype • Read-only evidence viewer • "
    "No training controls or experiment-output writes"
)
