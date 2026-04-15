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

Each MLflow experiment may contain runs from multiple design dimensions.
The design dimension is read from the `design_dimension` run tag and
results are shown per dimension.

Run with: marimo run rct_experiment_analysis.py
Edit with: marimo edit rct_experiment_analysis.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # RCT Experiment Analysis

        This notebook analyzes results from STGym RCT (Randomized Controlled Trial)
        experiments. It fetches data from MLflow, handles k-fold cross-validation
        aggregation, and computes within-group ranks for design choice comparison.

        Each experiment may contain runs from multiple design dimensions. Results
        are shown per design dimension, grouped automatically from the
        `design_dimension` run tag.
        """
    )
    return


@app.cell
def _():
    import warnings

    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    warnings.filterwarnings("ignore")

    return pd, plt, sns, warnings


@app.cell
def _():
    from stgym.rct_analysis import (
        analyze_experiment,
        summarize_ranks_by_design_choice,
    )

    return (
        analyze_experiment,
        summarize_ranks_by_design_choice,
    )


@app.cell
def _(mo):
    mo.md("## Configuration")
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
    metric_name_input = mo.ui.text(
        value="test_roc_auc",
        label="Metric Name",
    )
    metric_name_input
    return (metric_name_input,)


@app.cell
def _(mo):
    mo.md("## Load and Analyze Data")
    return


@app.cell
def _(
    analyze_experiment,
    experiment_id_input,
    metric_name_input,
    mo,
    tracking_uri_input,
):
    # Only run analysis if experiment_id is provided
    if experiment_id_input.value:
        try:
            results_by_dim = analyze_experiment(
                tracking_uri=tracking_uri_input.value,
                experiment_id=experiment_id_input.value,
                metric_name=metric_name_input.value,
                aggregate_kfold=True,
            )
            n_dims = len(results_by_dim)
            if n_dims == 0:
                analysis_status = mo.md(
                    f"*No runs found for experiment* `{experiment_id_input.value}`"
                )
            else:
                dim_list = ", ".join(f"`{d}`" for d in sorted(results_by_dim))
                analysis_status = mo.md(
                    f"Loaded **{n_dims}** design dimension(s) from experiment "
                    f"`{experiment_id_input.value}`: {dim_list}"
                )
        except Exception as e:
            results_by_dim = {}
            analysis_status = mo.md(f"**Error:** {e}")
    else:
        results_by_dim = {}
        analysis_status = mo.md("*Enter an experiment ID above to load data*")

    analysis_status
    return analysis_status, results_by_dim


@app.cell
def _(mo, results_by_dim, summarize_ranks_by_design_choice, plt, sns):
    # Render a section per design dimension
    if results_by_dim:
        sections = []
        for _dim, _df in sorted(results_by_dim.items()):
            sections.append(mo.md(f"## Design Dimension: `{_dim}`"))

            # Data preview
            sections.append(mo.md("### Data Preview"))
            sections.append(mo.ui.table(_df.head(20)))

            # Rank summary
            sections.append(mo.md("### Rank Summary by Design Choice"))
            _rank_summary = summarize_ranks_by_design_choice(_df)
            sections.append(mo.ui.table(_rank_summary))

            # Rank distribution plot
            sections.append(mo.md("### Rank Distribution"))
            _fig, _ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=_df, x="design_choice", y="rank", ax=_ax)
            _ax.set_xlabel(_dim)
            _ax.set_ylabel("Rank (1 = best)")
            _ax.set_title(f"Rank Distribution — {_dim}")
            plt.tight_layout()
            sections.append(_fig)

            # Metric distribution plot
            sections.append(mo.md("### Metric Distribution"))
            _fig2, _ax2 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=_df, x="design_choice", y="metric", ax=_ax2)
            _ax2.set_xlabel(_dim)
            _ax2.set_ylabel("metric")
            _ax2.set_title(f"Metric Distribution — {_dim}")
            plt.tight_layout()
            sections.append(_fig2)

        mo.vstack(sections)
    return


@app.cell
def _(mo, results_by_dim):
    if results_by_dim:
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
