import gc
import os
import traceback
from typing import Optional

import objgraph
import psutil

import torch
from logzero import logger as logz_logger
from omegaconf import OmegaConf
from pytorch_lightning.profilers import PyTorchProfiler

from stgym.config_schema import ExperimentConfig, MLFlowConfig, TaskConfig
from stgym.data_loader import STDataModule, STKfoldDataModule
from stgym.tl_model import STGymModule
from stgym.train import train
from stgym.types import primitive_type
from stgym.utils import gated_load, log_params_and_config_in_mlflow


def log_training_error(e: Exception, logger: Optional, error_context: str = ""):
    """Log training error to both console and MLflow if available."""
    error_msg = f"Training failed{error_context}: {e}"
    full_stacktrace = traceback.format_exc()
    logz_logger.error(error_msg)
    traceback.print_exc()
    if logger is not None:
        logger.experiment.log_text(
            logger.run_id,
            f"{error_msg}\n\nFull Stacktrace:\n{full_stacktrace}",
            "training_error.txt",
        )

        logger.experiment.set_tag(
            logger.run_id, "error", f"{type(e).__name__}: {str(e)}"
        )


def get_dim_out(task_cfg: TaskConfig) -> int | None:
    """Get output dimension of the GNN model based on experiment config"""
    if task_cfg.type in ("node-classification", "graph-classification"):
        dim_out = task_cfg.num_classes
        if dim_out == 2:
            dim_out -= 1  # for binary classification, output dim being 1 is enough
    else:
        raise ValueError(f"Unsupported task type: {task_cfg.type}")
    return dim_out


TL_TRAIN_CFG = {
    "log_every_n_steps": 5,
    # simplify the logging
    # "enable_progress_bar": True,
    # "enable_model_summary": True,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "enable_checkpointing": False,
}


def run_exp(
    exp_cfg: ExperimentConfig,
    mlflow_cfg: MLFlowConfig,
    metadata_for_tag: dict[str, primitive_type] | None = None,
    profile: bool = False,
    gated_datasets: frozenset[str] = frozenset(),
):
    _proc = psutil.Process(os.getpid())
    _rss_start_gb = _proc.memory_info().rss / 1e9
    logz_logger.info(f"[mem-diag] PID={_proc.pid} rss_start={_rss_start_gb:.2f} GB")
    objgraph.show_growth(limit=10)

    logz_logger.debug(OmegaConf.to_yaml(exp_cfg.model_dump()))

    dim_out = get_dim_out(exp_cfg.task)

    use_kfold_cv = exp_cfg.data_loader.use_kfold_split

    mlflow_cfg = mlflow_cfg.model_copy()
    mlflow_cfg.tags = {
        "task_type": exp_cfg.task.type,
        "dataset_name": exp_cfg.task.dataset_name,
    }
    if exp_cfg.group_id is not None:
        mlflow_cfg.tags |= {"group_id": str(exp_cfg.group_id)}
    if metadata_for_tag is not None:
        mlflow_cfg.tags |= metadata_for_tag

    print(f"mlflow_cfg: {mlflow_cfg}")

    profiler = None
    if profile:
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        profiler = PyTorchProfiler(
            dirpath="/tmp/stgym_profile",
            filename=exp_cfg.task.dataset_name,
            sort_by_key=sort_key,
            row_limit=25,
            export_to_chrome=True,
        )
        logz_logger.info("PyTorchProfiler enabled — traces → /tmp/stgym_profile/")

    if not use_kfold_cv:
        logz_logger.info("Evaluation mode: train/validation/test split.")
        # Create logger for single experiment
        if mlflow_cfg.track:
            logger = mlflow_cfg.create_tl_logger()
            log_params_and_config_in_mlflow(exp_cfg, logger)
        else:
            logger = None

        try:
            with gated_load(exp_cfg.task.dataset_name, gated_datasets):
                data_module = STDataModule(exp_cfg.task, exp_cfg.data_loader)
            model_module = STGymModule(
                dim_in=data_module.num_features,
                dim_out=dim_out,
                model_cfg=exp_cfg.model,
                train_cfg=exp_cfg.train,
                task_cfg=exp_cfg.task,
                dl_cfg=exp_cfg.data_loader,
            )
            train(
                model_module,
                data_module,
                exp_cfg.train,
                mlflow_cfg,
                tl_train_config=TL_TRAIN_CFG,
                logger=logger,
                profiler=profiler,
            )
        except Exception as e:
            log_training_error(e, logger)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                # Force-kill the Ray worker process so its CUDA context is destroyed
                # and the GPU slot is released. A normal raise only returns the worker
                # to the pool, permanently claiming the slot.
                # https://github.com/xiaohan2012/stgym/pull/99
                if logger is not None:
                    logger.experiment.set_terminated(logger.run_id, status="FAILED")
                os._exit(1)
    else:
        logz_logger.info("Evaluation mode: k-fold cross validation.")
        # k-fold split - create separate logger for each fold
        for fold in range(exp_cfg.data_loader.split.num_folds):
            exp_cfg.data_loader.split.split_index = fold
            # trigger model validation and post-processing logic
            exp_cfg = exp_cfg.validate()

            fold_dl_cfg = exp_cfg.data_loader.model_copy()
            # Explicitly set fold index to ensure correct data split
            fold_dl_cfg.split.split_index = fold

            # Create individual logger for this fold
            if mlflow_cfg.track:
                fold_logger = mlflow_cfg.create_tl_logger()
                fold_logger.experiment.set_tag(fold_logger.run_id, "fold", str(fold))
                log_params_and_config_in_mlflow(exp_cfg, fold_logger)
            else:
                fold_logger = None

            try:
                with gated_load(exp_cfg.task.dataset_name, gated_datasets):
                    fold_data_module = STKfoldDataModule(exp_cfg.task, fold_dl_cfg)

                # Create model module for this fold
                fold_model_module = STGymModule(
                    dim_in=fold_data_module.num_features,
                    dim_out=dim_out,
                    model_cfg=exp_cfg.model,
                    train_cfg=exp_cfg.train,
                    task_cfg=exp_cfg.task,
                    dl_cfg=fold_dl_cfg,
                )

                train(
                    fold_model_module,
                    fold_data_module,
                    exp_cfg.train,
                    mlflow_cfg,
                    tl_train_config=TL_TRAIN_CFG,
                    logger=fold_logger,
                    profiler=profiler,
                )
            except Exception as e:
                log_training_error(e, fold_logger, f" in fold {fold}")
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    # https://github.com/xiaohan2012/stgym/pull/99
                    if fold_logger is not None:
                        fold_logger.experiment.set_terminated(
                            fold_logger.run_id, status="FAILED"
                        )
                    os._exit(1)

    gc.collect()
    _rss_after_gc_gb = _proc.memory_info().rss / 1e9
    logz_logger.info(
        f"[mem-diag] PID={_proc.pid} "
        f"rss_after_gc={_rss_after_gc_gb:.2f} GB  "
        f"delta={_rss_after_gc_gb - _rss_start_gb:+.2f} GB"
    )
    objgraph.show_growth(limit=10)

    return True
