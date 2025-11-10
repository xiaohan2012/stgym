import os
import traceback

import ray
import torch
from logzero import logger as logz_logger
from omegaconf import OmegaConf

from stgym.config_schema import ExperimentConfig, MLFlowConfig, TaskConfig
from stgym.data_loader import STDataModule, STKfoldDataModule
from stgym.tl_model import STGymModule
from stgym.train import train
from stgym.utils import log_params_and_config_in_mlflow


def get_dim_out(task_cfg: TaskConfig) -> int | None:
    """Get output dimension of the GNN model based on experiment config"""
    if task_cfg.type in ("node-classification", "graph-classification"):
        dim_out = task_cfg.num_classes
        if dim_out == 2:
            dim_out -= 1  # for binary classification, output dim being 1 is enough
    else:
        # for clustering, dim_out is specified by the pooling operation
        dim_out = None
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


def run_exp(exp_cfg: ExperimentConfig, mlflow_cfg: MLFlowConfig):
    logz_logger.debug(OmegaConf.to_yaml(exp_cfg.model_dump()))
    logger = mlflow_cfg.create_tl_logger()

    # Ensure consistent CUDA device usage
    if torch.cuda.is_available():
        try:
            # Get Ray-assigned GPU devices
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                # Use the first assigned GPU
                device_id = gpu_ids[0]
                cuda_device = f"cuda:{device_id}"

                # Set CUDA_VISIBLE_DEVICES to restrict to assigned device
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

                # Update experiment config to use consistent device
                exp_cfg = exp_cfg.model_copy()
                exp_cfg.data_loader.device = cuda_device

                logz_logger.info(f"Using CUDA device: {cuda_device}")
            else:
                logz_logger.warning(
                    "No GPU assigned by Ray, falling back to default CUDA device"
                )
        except Exception as e:
            logz_logger.warning(
                f"Failed to set Ray GPU device: {e}, using default device"
            )

    dim_out = get_dim_out(exp_cfg.task)

    use_kfold_cv = exp_cfg.data_loader.use_kfold_split

    if exp_cfg.group_id is not None:
        mlflow_cfg = mlflow_cfg.model_copy()
        mlflow_cfg.tags = {
            "group_id": str(exp_cfg.group_id),
            "task_type": exp_cfg.task.type,
            "dataset_name": exp_cfg.task.dataset_name,
        }

    if logger is not None and mlflow_cfg.track:
        log_params_and_config_in_mlflow(exp_cfg, logger)

    try:
        if not use_kfold_cv:
            logz_logger.info("Evaluation mode: train/validation/test split.")
            # regular train/val/test split
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
            )
        else:
            logz_logger.info("Evaluation mode: k-fold cross validation.")
            # k-fold split
            for fold in range(exp_cfg.data_loader.split.num_folds):
                exp_cfg.data_loader.split.split_index = fold
                # trigger model validation and post-processing logic
                exp_cfg = exp_cfg.validate()

                fold_dl_cfg = exp_cfg.data_loader.model_copy()

                # Create data module for this fold
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
                    logger=logger,
                )

    except Exception as e:
        error_msg = f"Training failed: {e}"
        logz_logger.error(error_msg)
        traceback.print_exc()
        if logger is not None:
            logger.experiment.log_text(logger.run_id, error_msg, "training_error.txt")

    return True
