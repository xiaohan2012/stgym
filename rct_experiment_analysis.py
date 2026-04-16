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
results are shown per dimension automatically.

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

    Each experiment may contain runs from multiple design dimensions. Results
    are shown per design dimension, grouped automatically from the
    `design_dimension` run tag.
    """)
    return


@app.cell
def _():
    import warnings

    import seaborn as sns
    from matplotlib import pyplot as plt

    # Suppress specific warnings from seaborn/matplotlib during plotting
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    return plt, sns


@app.cell
def _():
    from stgym.rct_utils import (
        analyze_experiment,
        summarize_ranks_by_design_choice,
    )

    return (analyze_experiment,)


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
    metric_name_input = mo.ui.text(
        value="test_roc_auc",
        label="Metric Name",
    )
    metric_name_input
    return (metric_name_input,)


@app.cell
def _(experiment_id_input, mo):
    mo.stop(
        not experiment_id_input.value,
        mo.md("*Enter an experiment ID above to load data*"),
    )
    return


@app.cell
def _(
    analyze_experiment,
    experiment_id_input,
    metric_name_input,
    tracking_uri_input,
):
    results_df = analyze_experiment(
        tracking_uri=tracking_uri_input.value,
        experiment_id=experiment_id_input.value,
        metric_name=metric_name_input.value,
        aggregate_kfold=True,
    )
    return (results_df,)


@app.cell
def _(mo, results_df):
    mo.stop(
        results_df.empty,
        mo.md("*No data available. Check the experiment ID and MLflow connection.*"),
    )
    # Re-export so downstream cells depend on this gate and don't run when empty.
    active_df = results_df
    return (active_df,)


@app.cell
def _(active_df, mo):
    _dims = sorted(active_df["design_dimension"].unique())
    _dim_list = ", ".join(f"`{d}`" for d in _dims)
    mo.md(f"Loaded **{len(_dims)}** design dimension(s): {_dim_list}")
    return


@app.cell
def _(active_df, mo):
    rank_stats = (
        active_df.groupby(["design_dimension", "design_choice"])["rank"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values(["design_dimension", "mean"])
    )
    mo.vstack(
        [
            mo.md("## Rank Statistics by Design Choice"),
            mo.ui.table(rank_stats),
        ]
    )
    return (rank_stats,)


@app.cell
def _(active_df, mo, plt, sns):
    # Compute order by mean rank for each dimension
    _order_df = (
        active_df.groupby(["design_dimension", "design_choice"])["rank"]
        .mean()
        .reset_index()
    )
    _order_df = _order_df.sort_values(["design_dimension", "rank"])


    # Create ordered categorical for each dimension
    def _get_order(dim):
        return _order_df[_order_df["design_dimension"] == dim][
            "design_choice"
        ].tolist()


    sections = []

    sections.append(mo.md("### Rank Distribution"))
    _g = sns.FacetGrid(
        active_df, col="design_dimension", sharey=False, sharex=False, col_wrap=3
    )


    def _violin_with_order(data, **kwargs):
        dim = data["design_dimension"].iloc[0]
        order = _get_order(dim)
        sns.violinplot(
            data=data, x="design_choice", y="rank", order=order, **kwargs
        )


    _g.map_dataframe(_violin_with_order)
    _g.set_axis_labels("Design Choice", "Rank (1 = best)")
    _g.set_titles("{col_name}")
    plt.tight_layout()
    sections.append(_g.figure)

    sections.append(mo.md("### Metric Distribution"))
    _g2 = sns.FacetGrid(
        active_df, col="design_dimension", sharey=False, sharex=False, col_wrap=3
    )


    def _box_with_order(data, **kwargs):
        dim = data["design_dimension"].iloc[0]
        order = _get_order(dim)
        sns.boxplot(
            data=data, x="design_choice", y="metric", order=order, **kwargs
        )


    _g2.map_dataframe(_box_with_order)
    _g2.set_axis_labels("Design Choice", "Metric")
    _g2.set_titles("{col_name}")
    plt.tight_layout()
    sections.append(_g2.figure)

    mo.vstack(sections)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    - **Rank 1** = best performing design choice within each group
    - Lower mean rank indicates a design choice that consistently outperforms alternatives
    - Compare the rank distributions to assess consistency vs variability
    """)
    return


@app.cell(hide_code=True)
def _(mo, rank_stats):
    import os

    rank_stats.to_excel(os.path.expanduser("~/Desktop/rank_stats.xlsx"))
    mo.md("Exported to `~/Desktop/rank_stats.xlsx`")
    return


if __name__ == "__main__":
    app.run()
