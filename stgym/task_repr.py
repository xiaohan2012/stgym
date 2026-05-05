from dataclasses import dataclass

import pandas as pd
from logzero import logger
from mlflow.entities import Run

from stgym.config_schema import (
    DataLoaderConfig,
    GraphClassifierModelConfig,
    NodeClassifierModelConfig,
    TaskType,
    TrainConfig,
    dataset_eval_mode,
)
from stgym.design_space.design_gen import (
    generate_data_loader_config,
    generate_model_config,
    generate_train_config,
)
from stgym.design_space.schema import TaskReprDesignSpace

ModelConfig = GraphClassifierModelConfig | NodeClassifierModelConfig

TASK_METRIC: dict[str, str] = {
    "graph-classification": "test_roc_auc",
    "node-classification": "test_accuracy",
}


@dataclass
class TaskFreeDesign:
    design_id: int
    model: ModelConfig
    train: TrainConfig
    data_loader: DataLoaderConfig


def expected_mlflow_run_count(n_designs: int, tasks: list[str]) -> int:
    """Return the number of MLflow runs a task-repr sweep will produce.

    k-fold datasets create one run per fold; all others create one run.
    """
    runs_per_design = sum(
        dataset_eval_mode[t].num_folds if t in dataset_eval_mode else 1 for t in tasks
    )
    return n_designs * runs_per_design


def build_performance_matrix(
    runs: list[Run],
    task_type: TaskType,
) -> pd.DataFrame:
    """Build a design × task performance matrix from MLflow runs.

    Rows are design_ids (int), columns are dataset names. k-fold runs for the
    same (design_id, dataset_name) pair are averaged. Missing cells are NaN.
    """
    metric = TASK_METRIC[task_type]
    records = []
    for run in runs:
        design_id = int(run.data.tags["design_id"])
        dataset_name = run.data.tags["dataset_name"]
        value = run.data.metrics.get(metric)
        if value is not None:
            records.append(
                {"design_id": design_id, "dataset_name": dataset_name, "value": value}
            )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # k-fold runs for the same (design_id, dataset_name) cell are averaged
    return (
        df.groupby(["design_id", "dataset_name"])["value"]
        .mean()
        .unstack("dataset_name")
    )


def select_anchor_models(
    matrix: pd.DataFrame,
    n_anchors: int,
) -> list[int]:
    """Select anchor model design_ids from a performance matrix.

    Drops designs with any NaN, ranks remainder by mean score across tasks,
    divides into n_anchors equal bins, and picks the design at index n//2
    within each bin (upper-mid for even bin sizes).
    """
    dropped_cols = matrix.columns[matrix.isna().all()].tolist()
    if dropped_cols:
        logger.warning(f"Dropping all-NaN datasets from matrix: {dropped_cols}")
    trimmed = matrix.drop(columns=dropped_cols)
    means = trimmed.mean(axis=1, skipna=True)
    valid = means.dropna()
    n_excluded = len(matrix) - len(valid)
    logger.info(
        f"Anchor selection: {len(valid)} valid designs, {n_excluded} excluded (all-NaN)"
    )
    if len(valid) < n_anchors:
        raise ValueError(
            f"Only {len(valid)} complete designs after dropping NaN rows, "
            f"need at least {n_anchors}."
        )
    ranked = valid.sort_values()
    design_ids = ranked.index.tolist()
    bin_size = len(design_ids) // n_anchors
    return [design_ids[i * bin_size + bin_size // 2] for i in range(n_anchors)]


def sample_task_free_designs(
    task_type: TaskType,
    space: TaskReprDesignSpace,
    n_designs: int,
    seed: int,
) -> list[TaskFreeDesign]:
    model_configs = generate_model_config(task_type, space.model, n_designs, seed)
    train_configs = generate_train_config(space.train, n_designs, seed)
    dl_configs = generate_data_loader_config(space.data_loader, n_designs, seed)
    return [
        TaskFreeDesign(design_id=i, model=m, train=tr, data_loader=dl)
        for i, (m, tr, dl) in enumerate(zip(model_configs, train_configs, dl_configs))
    ]
