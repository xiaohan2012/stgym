#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "mlflow",
#     "pandas",
#     "pydash",
#     "seaborn",
#     "matplotlib",
# ]
# ///
"""
RCT Experiment Analysis

Interactive Marimo notebook for analyzing STGym RCT experiment results.
Loads data from MLflow, aggregates k-fold CV results, computes ranks,
and visualizes design choice comparisons.

Run with: marimo run rct_experiment_analysis.py
Edit with: marimo edit rct_experiment_analysis.py
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # RCT Experiment Analysis

    This notebook analyzes results from STGym RCT (Randomized Controlled Trial)
    experiments. It fetches data from MLflow, handles k-fold cross-validation
    aggregation, and computes within-group ranks for design choice comparison.
    """)
    return


@app.cell
def _():
    import warnings

    import seaborn as sns
    from matplotlib import pyplot as plt

    warnings.filterwarnings("ignore")
    return plt, sns


@app.cell
def _():
    from stgym.rct_analysis import (
        analyze_experiment,
        detect_design_dimension,
        summarize_ranks_by_design_choice,
    )

    return (
        analyze_experiment,
        detect_design_dimension,
        summarize_ranks_by_design_choice,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    tracking_uri_input = mo.ui.text(
        value="http://127.0.0.1:5001",
        label="MLflow Tracking URI",
    )
    tracking_uri_input
    return (tracking_uri_input,)


@app.cell
def _(mo):
    experiment_id_input = mo.ui.text(
        value="",
        label="Experiment ID",
        placeholder="Enter MLflow experiment ID",
    )
    experiment_id_input
    return (experiment_id_input,)


@app.cell
def _(mo):
    experiment_name_dropdown = mo.ui.dropdown(
        options=[
            "bn",
            "hpooling",
            "activation",
            "lr",
            "optimizer",
            "batch_size",
            "epochs",
            "clusters",
            "knn",
            "radius",
            "n_mlp_layers",
            "mlp_dim_inner",
            "postmp",
            "(custom)",
        ],
        value="hpooling",
        label="Design Dimension Preset",
    )
    experiment_name_dropdown
    return (experiment_name_dropdown,)


@app.cell
def _(detect_design_dimension, experiment_name_dropdown, mo):
    # Get preset configuration or show custom inputs
    preset_config = detect_design_dimension(experiment_name_dropdown.value)

    if preset_config:
        design_dimension = preset_config["design_dimension"]
        metric_name = preset_config["metric_name"]
        design_label = preset_config["design_choice_label"]
        mo.md(
            f"""
            **Using preset:** {experiment_name_dropdown.value}

            - Design dimension: `{design_dimension}`
            - Metric: `{metric_name}`
            - Label: `{design_label}`
            """
        )
    else:
        design_dimension = "model/mp_layers/0/pooling/type"
        metric_name = "test_roc_auc"
        design_label = "design_choice"
    return design_dimension, design_label, metric_name


@app.cell
def _(mo):
    mo.md("""
    ## Load and Analyze Data
    """)
    return


@app.cell
def _(
    analyze_experiment,
    design_dimension,
    experiment_id_input,
    metric_name,
    tracking_uri_input,
):
    # Only run analysis if experiment_id is provided
    assert experiment_id_input.value is not None

    df = analyze_experiment(
        tracking_uri=tracking_uri_input.value,
        experiment_id=experiment_id_input.value,
        design_dimension=design_dimension,
        metric_name=metric_name,
        aggregate_kfold=True,
    )

    return (df,)


@app.cell
def _(df, mo):
    if df is not None and not df.empty:
        mo.md("## Data Preview")
    return


@app.cell
def _(df, mo):
    if df is not None and not df.empty:
        mo.ui.table(df.head(20))
    return


@app.cell
def _(df, mo):
    if df is not None and not df.empty:
        mo.md("## Rank Summary by Design Choice")
    return


@app.cell
def _(df, mo, summarize_ranks_by_design_choice):
    if df is not None and not df.empty:
        rank_summary = summarize_ranks_by_design_choice(df)
        mo.ui.table(rank_summary)
    return


@app.cell
def _(df, mo):
    if df is not None and not df.empty:
        mo.md("## Rank Distribution Visualization")
    return


@app.cell
def _(design_label, df, plt, sns):
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x="design_choice", y="rank", ax=ax)
        ax.set_xlabel(design_label)
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title("Rank Distribution by Design Choice")
        plt.tight_layout()
        fig
    return


@app.cell
def _(df, mo):
    if df is not None and not df.empty:
        mo.md("## Metric Distribution")
    return


@app.cell
def _(design_label, df, metric_name, plt, sns):
    if df is not None and not df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="design_choice", y="metric", ax=ax2)
        ax2.set_xlabel(design_label)
        ax2.set_ylabel(metric_name)
        ax2.set_title(f"{metric_name} Distribution by Design Choice")
        plt.tight_layout()
        fig2
    return


@app.cell
def _(df, mo):
    if df is not None and not df.empty:
        mo.md(
            """
            ## Interpretation

            - **Rank 1** = best performing design choice within each group
            - Lower mean rank indicates a design choice that consistently outperforms alternatives
            - Compare the rank distributions to assess consistency vs variability
            """
        )
    return


if __name__ == "__main__":
    app.run()
