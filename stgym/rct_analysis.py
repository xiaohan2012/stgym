"""Utility functions for analyzing RCT (Randomized Controlled Trial) experiment results.

This module provides reusable functions for:
- Loading and preprocessing MLflow run data
- Aggregating k-fold cross-validation results
- Computing within-group ranks for design choice comparison
- Detecting design dimensions from experiment configurations
"""

from typing import Any

import pandas as pd
import pydash as _
from mlflow import MlflowClient
from mlflow.entities import Run

# Maps design dimension tag values (stored in run.data.tags.design_dimension)
# to the actual MLflow param paths used for that dimension.
DESIGN_DIM_TO_MLFLOW_PATH: dict[str, str] = {
    "model.act": "model/mp_layers/0/act",
    "model.use_batchnorm": "model/mp_layers/0/use_batchnorm",
    "model.dim_inner": "model/mp_layers/0/dim_inner",
    "model.layer_type": "model/mp_layers/0/layer_type",
    "model.pooling.type": "model/mp_layers/0/pooling/type",
    "model.pooling.n_clusters": "model/mp_layers/0/pooling/n_clusters",
    "model.global_pooling": "model/global_pooling",
    "model.normalize_adj": "model/normalize_adj",
    "model.post_mp_dims": "model/post_mp_layer/dims",
    "train.max_epoch": "train/max_epoch",
    "train.optim.base_lr": "train/optim/base_lr",
    "train.optim.optimizer": "train/optim/optimizer",
    "data_loader.batch_size": "data_loader/batch_size",
    "data_loader.knn_k": "data_loader/knn_k",
    "data_loader.radius_ratio": "data_loader/radius_ratio",
}


def fetch_runs(
    tracking_uri: str,
    experiment_id: str,
    max_results: int = 10000,
) -> list[Run]:
    """Fetch all runs from an MLflow experiment.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_id: The experiment ID to fetch runs from.
        max_results: Maximum number of runs to retrieve.

    Returns:
        List of MLflow Run objects.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    return client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_results,
    )


def runs_to_dataframe(
    runs: list[Run],
    metric_name: str,
) -> pd.DataFrame:
    """Convert MLflow runs to a DataFrame with essential columns.

    Extracts group_id, fold, metric value, design choice, design_dimension,
    and run status from each run. The design_dimension is read from
    ``run.data.tags.design_dimension`` and the corresponding param value is
    looked up via :data:`DESIGN_DIM_TO_MLFLOW_PATH`.

    Args:
        runs: List of MLflow Run objects.
        metric_name: Name of the metric to extract (e.g., "test_roc_auc").

    Returns:
        DataFrame with columns: group_id, fold, metric, design_choice,
        design_dimension, run_status.
    """
    metric_path = f"data.metrics.{metric_name}"

    def _run_to_row(r: Run) -> dict[str, Any]:
        design_dim = _.get(r, "data.tags.design_dimension")
        mlflow_path = (
            DESIGN_DIM_TO_MLFLOW_PATH.get(design_dim, "") if design_dim else ""
        )
        design_choice = _.get(r, f"data.params.{mlflow_path}") if mlflow_path else None
        return {
            "group_id": _.get(r, "data.tags.group_id"),
            "fold": _.get(r, "data.tags.fold"),
            "metric": _.get(r, metric_path),
            "design_choice": design_choice,
            "design_dimension": design_dim,
            "run_status": _.get(r, "info.status"),
        }

    return pd.DataFrame(_.map_(runs, _run_to_row))


def aggregate_kfold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate k-fold cross-validation results by taking the mean metric per group.

    For experiments using k-fold CV, each configuration (group_id + design_choice)
    has multiple folds. This function computes the mean metric across folds.

    Args:
        df: DataFrame with columns including group_id, design_choice, metric, fold.

    Returns:
        DataFrame with one row per (group_id, design_choice) pair, containing
        the mean metric value. Non-k-fold runs (fold=None) are passed through.
    """
    # Separate k-fold and non-k-fold runs
    has_fold = df["fold"].notna()

    if not has_fold.any():
        # No k-fold runs, return as-is
        return df.copy()

    kfold_df = df[has_fold].copy()
    non_kfold_df = df[~has_fold].copy()

    # Aggregate k-fold runs: mean metric per (group_id, design_choice)
    aggregated = (
        kfold_df.groupby(["group_id", "design_choice"], as_index=False)
        .agg(
            {
                "metric": "mean",
                "run_status": lambda x: (
                    "FINISHED" if (x == "FINISHED").all() else "FAILED"
                ),
            }
        )
        .assign(fold=None)
    )

    return pd.concat([non_kfold_df, aggregated], ignore_index=True)


def filter_complete_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out groups where any run failed.

    Only keeps groups where all runs have status FINISHED.

    Args:
        df: DataFrame with columns including group_id, run_status.

    Returns:
        DataFrame containing only complete groups.
    """
    return df.groupby("group_id").filter(
        lambda x: (x["run_status"] == "FINISHED").all()
    )


def compute_within_group_ranks(
    df: pd.DataFrame,
    ascending: bool = False,
) -> pd.DataFrame:
    """Compute within-group ranks for the metric column.

    Ranks are computed within each group_id. By default, higher metric = better
    (rank 1 is the highest).

    Args:
        df: DataFrame with columns including group_id, metric.
        ascending: If True, lower metric values get better ranks.
            Default False (higher is better).

    Returns:
        DataFrame with an additional "rank" column.
    """
    df = df.copy()
    df["rank"] = df.groupby("group_id")["metric"].rank(ascending=ascending)
    return df


def summarize_ranks_by_design_choice(
    df: pd.DataFrame,
    design_choice_col: str = "design_choice",
) -> pd.DataFrame:
    """Summarize rank statistics by design choice.

    Computes mean rank, count, standard deviation, and median rank
    for each design choice.

    Args:
        df: DataFrame with columns including design_choice, rank.
        design_choice_col: Name of the design choice column.

    Returns:
        DataFrame with summary statistics per design choice.
    """
    return (
        df.groupby(design_choice_col)["rank"]
        .agg(["mean", "count", "std", "median"])
        .reset_index()
    )


def detect_design_dimension(experiment_name: str) -> dict[str, Any] | None:
    """Detect design dimension configuration from experiment name.

    Maps known experiment names to their design dimension paths and
    metric configurations.

    Args:
        experiment_name: Name of the MLflow experiment (e.g., "bn", "hpooling").

    Returns:
        Dict with keys: design_dimension, metric_name, design_choice_label.
        Returns None if experiment name is not recognized.
    """
    # Mapping: experiment_name -> (design_dimension, design_choice_label)
    # All experiments currently use test_roc_auc as the metric
    dimension_map: dict[str, tuple[str, str]] = {
        "bn": ("model/post_mp_layer/use_batchnorm", "use_batchnorm"),
        "hpooling": ("model/mp_layers/0/pooling/type", "hpooling_type"),
        "activation": ("model/mp_layers/0/act", "activation"),
        "lr": ("train/optim/base_lr", "learning_rate"),
        "optimizer": ("train/optim/optimizer", "optimizer"),
        "batch_size": ("data_loader/batch_size", "batch_size"),
        "epochs": ("train/max_epoch", "max_epoch"),
        "clusters": ("model/mp_layers/0/pooling/n_clusters", "n_clusters"),
        "knn": ("data_loader/knn_k", "knn_k"),
        "radius": ("data_loader/radius_ratio", "radius_ratio"),
        "n_mlp_layers": ("model/post_mp_layer/n_layers", "n_mlp_layers"),
        "mlp_dim_inner": ("model/post_mp_layer/dim_inner", "mlp_dim_inner"),
        "postmp": ("model/post_mp_layer/type", "postmp_type"),
    }

    if experiment_name not in dimension_map:
        return None

    design_dimension, design_choice_label = dimension_map[experiment_name]
    return {
        "design_dimension": design_dimension,
        "metric_name": "test_roc_auc",
        "design_choice_label": design_choice_label,
    }


def analyze_experiment(
    tracking_uri: str,
    experiment_id: str,
    metric_name: str = "test_roc_auc",
    aggregate_kfold: bool = True,
) -> dict[str, pd.DataFrame]:
    """High-level function to analyze an RCT experiment.

    Fetches runs, groups them by ``design_dimension`` tag, then for each
    dimension: converts to DataFrame, optionally aggregates k-fold results,
    filters incomplete groups, and computes ranks.

    An experiment may contain runs from multiple design dimensions; each is
    analyzed independently.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_id: The experiment ID to analyze.
        metric_name: Name of the metric to analyze.
        aggregate_kfold: If True, aggregate k-fold results before ranking.

    Returns:
        Dict mapping each design dimension name to a DataFrame with columns:
        group_id, design_choice, metric, rank, and run_status.
        Returns an empty dict when there are no runs.
    """
    runs = fetch_runs(tracking_uri, experiment_id)

    if not runs:
        return {}

    df = runs_to_dataframe(runs, metric_name)

    results: dict[str, pd.DataFrame] = {}
    for design_dim, dim_df in df.groupby("design_dimension"):
        dim_df = dim_df.copy()
        if aggregate_kfold:
            dim_df = aggregate_kfold_metrics(dim_df)
        dim_df = filter_complete_groups(dim_df)
        dim_df = compute_within_group_ranks(dim_df)
        results[str(design_dim)] = dim_df

    return results
