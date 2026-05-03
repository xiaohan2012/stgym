"""Task representation via anchor-model performance fingerprinting.

Pipeline:
  1. sample_designs        — sample D task-free configs from a TaskReprDesignSpace
  2. (run experiments)     — run_task_repr.py dispatches D×T Ray tasks, tags each with design_id
  3. build_performance_matrix — pivot MLflow runs into a design_id × dataset DataFrame
  4. select_anchor_models  — bin designs by avg score, pick median per bin
  5. compute_fingerprints  — subset matrix to anchor rows, transpose to task × anchor
  6. pairwise_similarity   — Kendall tau between every pair of task fingerprint vectors
"""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from stgym.config_schema import (
    ExperimentConfig,
    TaskConfig,
)
from stgym.data_loader.ds_info import get_info
from stgym.design_space.design_gen import (
    generate_data_loader_config,
    generate_model_config,
    generate_train_config,
)
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.utils import rand_ints


def sample_designs(
    space: TaskReprDesignSpace,
    task_type: str,
    n: int,
    seed: int = 42,
) -> list[tuple[int, dict]]:
    """Sample n task-free design configs from a TaskReprDesignSpace.

    Returns a list of (design_id, partial_config_dict) where partial_config_dict
    contains model, train, and data_loader keys but no task. design_id is a
    stable integer index 0..n-1.
    """
    seeds = rand_ints(n, seed=seed)
    model_cfgs = generate_model_config(task_type, space.model, k=n, seed=seed)
    train_cfgs = generate_train_config(space.train, k=n, seed=seed)
    dl_cfgs = generate_data_loader_config(space.data_loader, k=n, seed=seed)

    return [
        (i, {"model": model_cfgs[i], "train": train_cfgs[i], "data_loader": dl_cfgs[i]})
        for i in range(n)
    ]


def make_exp_config(
    partial: dict,
    dataset_name: str,
) -> ExperimentConfig:
    """Attach a task to a partial design config to produce a full ExperimentConfig."""
    ds_info = get_info(dataset_name)
    task_cfg = TaskConfig(
        dataset_name=dataset_name,
        type=ds_info["task_type"],
        num_classes=ds_info["num_classes"],
    )
    return ExperimentConfig(
        model=partial["model"],
        train=partial["train"],
        data_loader=partial["data_loader"],
        task=task_cfg,
    )


def build_performance_matrix(
    runs: list,
    metric_name: str,
    min_task_coverage: float = 0.5,
) -> pd.DataFrame:
    """Build a design_id × dataset performance matrix from MLflow runs.

    Runs must have the 'design_id' tag set by run_task_repr.py. K-fold runs
    are averaged per (design_id, dataset_name) before pivoting.

    Args:
        runs: list of mlflow.entities.Run objects.
        metric_name: MLflow metric key to extract (e.g. 'test_roc_auc').
        min_task_coverage: drop designs that cover fewer than this fraction of tasks.

    Returns:
        DataFrame with index=design_id (int), columns=dataset_name, values=mean metric.
        Rows with insufficient task coverage are dropped.
    """
    rows = []
    for r in runs:
        design_id = r.data.tags.get("design_id")
        dataset = r.data.tags.get("dataset_name")
        fold = r.data.tags.get("fold")
        metric = r.data.metrics.get(metric_name)
        status = r.info.status
        if design_id is None or dataset is None or metric is None:
            continue
        rows.append(
            {
                "design_id": int(design_id),
                "dataset_name": dataset,
                "fold": fold,
                "metric": metric,
                "run_status": status,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # aggregate k-fold folds: mean metric, FINISHED only if all folds finished
    has_fold = df["fold"].notna()
    kfold_df = df[has_fold]
    non_kfold_df = df[~has_fold]

    parts = [non_kfold_df[["design_id", "dataset_name", "metric", "run_status"]]]
    if not kfold_df.empty:
        kfold_agg = kfold_df.groupby(["design_id", "dataset_name"], as_index=False).agg(
            metric=("metric", "mean"),
            run_status=(
                "run_status",
                lambda x: "FINISHED" if (x == "FINISHED").all() else "FAILED",
            ),
        )
        parts.append(kfold_agg)

    df = pd.concat(parts, ignore_index=True)
    df = df[df["run_status"] == "FINISHED"]

    # average across any remaining duplicates (same design re-run on same task)
    df = df.groupby(["design_id", "dataset_name"], as_index=False)["metric"].mean()

    matrix = df.pivot(index="design_id", columns="dataset_name", values="metric")

    n_tasks = matrix.shape[1]
    coverage = matrix.notna().sum(axis=1) / n_tasks
    matrix = matrix[coverage >= min_task_coverage]

    return matrix


def select_anchor_models(
    matrix: pd.DataFrame,
    n_anchors: int = 12,
) -> list[int]:
    """Select anchor models by binning designs on avg performance and taking medians.

    Designs are ranked by their mean score across all tasks, split into
    n_anchors equal-sized bins, and the design with the median score in each
    bin is selected.

    Args:
        matrix: design_id × dataset DataFrame (from build_performance_matrix).
        n_anchors: number of anchor models to select.

    Returns:
        List of design_id integers identifying the anchor models.
    """
    if len(matrix) < n_anchors:
        raise ValueError(
            f"Not enough designs ({len(matrix)}) to select {n_anchors} anchors."
        )

    avg_scores = matrix.mean(axis=1).sort_values()
    design_ids = avg_scores.index.tolist()
    bins = np.array_split(design_ids, n_anchors)

    anchors = []
    for bin_ids in bins:
        bin_scores = avg_scores.loc[bin_ids]
        median_idx = (bin_scores - bin_scores.median()).abs().idxmin()
        anchors.append(int(median_idx))

    return anchors


def compute_fingerprints(
    matrix: pd.DataFrame,
    anchor_ids: list[int],
) -> pd.DataFrame:
    """Compute task fingerprints as anchor-model performance vectors.

    Args:
        matrix: design_id × dataset DataFrame.
        anchor_ids: design_ids to use as anchor models.

    Returns:
        DataFrame with index=dataset_name, columns=anchor design_id,
        values=performance score of that anchor on that task.
    """
    anchor_rows = matrix.loc[anchor_ids]
    return anchor_rows.T


def pairwise_similarity(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise task similarity via Kendall rank correlation.

    Args:
        fingerprints: task × anchor_model DataFrame (from compute_fingerprints).

    Returns:
        Symmetric task × task DataFrame of Kendall tau correlation values.
    """
    tasks = fingerprints.index.tolist()
    n = len(tasks)
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            tau, _ = kendalltau(fingerprints.iloc[i], fingerprints.iloc[j])
            sim[i, j] = tau
            sim[j, i] = tau

    return pd.DataFrame(sim, index=tasks, columns=tasks)


def design_hash(partial: dict) -> str:
    """Stable hex digest of a partial design config (model + train + data_loader)."""
    serialisable = {
        "model": partial["model"].model_dump(),
        "train": partial["train"].model_dump(),
        "data_loader": partial["data_loader"].model_dump(),
    }
    s = json.dumps(serialisable, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]
